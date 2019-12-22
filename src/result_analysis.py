import json
import matplotlib.pyplot as plt


def get_data(path):
    with open(path, 'r') as f:
        data = json.load(f)

    time_steps = data['battle_won_mean_T']
    win_rate = data['test_battle_won_mean']
    ep_length_mean = data['test_ep_length_mean']
    return_mean = data['test_return_mean']

    return time_steps, win_rate, return_mean, ep_length_mean


def plot(x, qmix_data, vbc_data, title):
    assert len(qmix_data) == len(vbc_data)

    plt.plot(x, qmix_data, label='QMIX')
    plt.plot(x, vbc_data, label='VBC+QMIX')
    plt.legend(loc='best')
    plt.ylabel("{}".format(title))
    plt.xlabel("Training Episodes")
    plt.show()
    plt.close()


path = '../results/sacred/40_finished/info.json'
time_steps, win, reward, length = get_data(path)
plot(time_steps, win, win, 'Win')




