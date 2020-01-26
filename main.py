from cem import sepCEM
import ray,copy,torch,time
import torch.nn as nn
import statistics
import numpy as np
from copy import deepcopy



USE_CUDA = torch.cuda.is_available()




def ray_get_and_free(object_ids):
    """Call ray.get and then queue the object ids for deletion.

    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.

    Args:
        object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.

    Returns:
        The result of ray.get(object_ids).
    """

    global _last_free_time
    global _to_free

    result = ray.get(object_ids)
    # print("ray_get_and_free,object_ids",object_ids)

    # print("ray_get_and_free,result",result)

    if type(object_ids) is not list:
        object_ids = [object_ids]
    _to_free.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_to_free) > MAX_FREE_QUEUE_SIZE
            or now - _last_free_time > FREE_DELAY_S):
        ray.internal.free(_to_free)
        _to_free = []
        _last_free_time = now

    return result


class MLP(nn.Module):
    def __init__(self,
                layers,
                activation=torch.tanh,
                output_activation=None,
                output_scale=1,
                output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale
        return x.squeeze() if self.output_squeeze else x


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()




class RLNN(nn.Module):

    def __init__(self):
        super(RLNN, self).__init__()
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.max_action = max_action

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        """
        Loads the model
        """
        if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/{}.pkl'.format(filename, net_name),
                       map_location=lambda storage, loc: storage)
        )

    def save_model(self, output, net_name):
        """
        Saves the model
        """
        torch.save(
            self.state_dict(),
            '{}/{}.pkl'.format(output, net_name)
        )



class function_B(RLNN):
    def __init__(self, in_features, hidden_sizes, activation):
        super(function_B, self).__init__()
        self.action_dim = 1 # action_space.shape[0]
        # self.output_activation = output_activation
        self.mean = None
        self.std = None
        self.values = []
        # self.mean_std()

        self.net = MLP(
            layers=[in_features] + list(hidden_sizes),
            activation=activation,
            output_activation=activation)

        self.mu = nn.Linear(
            in_features=list(hidden_sizes)[-1], out_features=self.action_dim)

    def forward(self, x):
        output = self.net(torch.Tensor([x]))
        mu = self.mu(output)
        # mu = torch.tanh(mu)
        return mu.cpu().detach().numpy()[0]

    def mean_std(self):
        for x in range(-1000,1001):
            y = self.forward(x)
            self.values.append(y)
        self.mean = statistics.mean(self.values)
        self.std = statistics.pstdev(self.values)




def _calucalue_z_test(function_A,function_B):
    function_B.mean_std()

    z = (function_A.mean-function_B.mean)/math.sqrt(function_A.std+function_B.std)
    return z


# input x, output y
class function_A(object):
    def __init__(self):
        super(function_A, self).__init__()
        self.x_range = [-1000,1000]
        self.mean = None
        self.std = None
        self.values = []
        self.mean_std()

    def calculate(self,x):
        # return np.log(x)
        return pow(x,2)

    def mean_std(self):
        for x in range(-1000,1001):
            y = self.calculate(x)
            self.values.append(y)
        self.mean = statistics.mean(self.values)
        self.std = statistics.pstdev(self.values)


@ray.remote
class Engine(object):
    def __init__(self,args):
        self.args = args

        self.actor = function_B(1,(256, 256), torch.relu)
        self.es = sepCEM(self.actor.get_size(), mu_init=self.actor.get_params(), sigma_init=args.sigma_init, damp=args.damp, damp_limit=args.damp_limit,
        pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2, elitism=args.elitism)

    def calucalue_fitness(self,function_A):
        explorer_list_ids = []
        experiences_episode = []
        self.all_fitness = []
        timesteps_one_gen = 0
        self.es_params = self.es.ask(self.args.pop_size)

        for params in self.es_params:
            self.actor.set_params(params)
            z = _calucalue_z_test(function_A,self.actor)
            self.all_fitness.append(z)

        return self.all_fitness

    def evolve(self):
        self.es.tell(self.es_params, self.all_fitness)

    def evaluate_actor(self,function_A):
        wrong_number = 0
        for x in range(-1000,1001):
            y_a = function_A.calculate(x)
            y_b = self.actor(x)
            if abs(y_a-y_b) > 0.0001*(abs(y_a)+abs(y_b)):
                wrong_number += 1

        return wrong_number/2001



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop_size', type=int, default=20)
    # CEM
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)
    parser.add_argument('--elitism', dest="elitism",  action='store_true') # defult False

    args = parser.parse_args()

    ray.init(include_webui=False, ignore_reinit_error=True, object_store_memory=10000000000,memory=10000000000)#10000000000,memory=10000000000)



    engine = Engine.remote(args)
    timesteps = 0
    function_A = function_A()

    while True:
        ray_get_and_free(engine.calucalue_fitness.remote(function_A))
        ray_get_and_free(engine.evolve.remote())
        elite_fitness = ray_get_and_free(evaluate_actor.remote(function_A))
        if elite_fitness < 0.0001:
            break

        timesteps += 1














    





















print("test")