import os
import cPickle as pickle


def print_costs(load_params_file):
    params, v_grads, s_grads, costs, iteration, start_epoch = pickle.load(open(load_params_file, "rb"))

    costs = [cost[0] for cost in costs]

    epoch = 0
    for cost in costs:
        print("epoch = %d iteration = %d cost = %f" % (epoch, (epoch+1)*125, cost))
        epoch += 1


print_costs(os.path.join("../runs/44", "training_25.txt"))
print_costs(os.path.join("../runs/40", "training_5.txt"))
