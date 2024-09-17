import numpy as np
from sklearn.linear_model import LogisticRegression
from os.path import join
import quapy as qp
from quapy.protocol import UPP
from quapy.method.aggregative import KDEyML

DEBUG = True

qp.environ["SAMPLE_SIZE"] = 100 if DEBUG else 500
val_repeats  = 100 if DEBUG else 500
test_repeats = 100 if DEBUG else 500
if DEBUG:
    qp.environ["DEFAULT_CLS"] = LogisticRegression()

test_results = {}
val_choice = {}

bandwidth_range = np.linspace(0.01, 0.20, 20)
if DEBUG:
    bandwidth_range = np.linspace(0.01, 0.20, 10)

def datasets():
    for dataset_name in qp.datasets.UCI_MULTICLASS_DATASETS[:4]:
        dataset = qp.datasets.fetch_UCIMulticlassDataset(dataset_name)
        if DEBUG:
            dataset = dataset.reduce(random_state=0)
        yield dataset


def experiment_dataset(dataset):
    train, test = dataset.train_test
    test_gen = UPP(test, repeats=test_repeats)

    # bandwidth chosen during model selection in validation
    train_tr, train_va = train.split_stratified(random_state=0)
    kdey = KDEyML(random_state=0)
    modsel = qp.model_selection.GridSearchQ(
        model=kdey,
        param_grid={'bandwidth': bandwidth_range},
        protocol=UPP(train_va, repeats=val_repeats),
        refit=False,
        n_jobs=-1
    ).fit(train_tr)
    chosen_bandwidth = modsel.best_params_['bandwidth']
    modsel_choice = float(chosen_bandwidth)

    # results in test
    print(f"testing KDEy in {dataset.name}")
    dataset_results = []
    for b in bandwidth_range:
        kdey = KDEyML(bandwidth=b, random_state=0)
        kdey.fit(train)

        mae = qp.evaluation.evaluate(kdey, protocol=test_gen, error_metric='mae', verbose=True)
        print(f'bandwidth={b}: {mae:.5f}')
        dataset_results.append((float(b), float(mae)))

    return modsel_choice, dataset_results

def plot_bandwidth(val_choice, test_results):
    for dataset_name in val_choice.keys():
        import matplotlib.pyplot as plt

        bandwidths, results = zip(*test_results[dataset_name])

        # Crear la gráfica
        plt.figure(figsize=(8, 6))

        # Graficar los puntos de datos
        plt.plot(bandwidths, results, marker='o')

        # Agregar la línea vertical en bandwidth_chosen
        plt.axvline(x=val_choice[dataset_name], color='r', linestyle='--', label=f'Bandwidth elegido: {val_choice[dataset_name]}')

        # Agregar etiquetas y título
        plt.xlabel('Bandwidth')
        plt.ylabel('Resultado')
        plt.title('Gráfica de Bandwidth vs Resultado')

        # Mostrar la leyenda
        plt.legend()

        # Mostrar la gráfica
        plt.grid(True)
        plt.show()


for dataset in datasets():
    if DEBUG:
        result_path = f'./results/debug/{dataset.name}.pkl'
    else:
        result_path = f'./results/{dataset.name}.pkl'

    modsel_choice, dataset_results = qp.util.pickled_resource(result_path, experiment_dataset, dataset)
    val_choice[dataset.name] = modsel_choice
    test_results[dataset.name] = dataset_results

    print(f'Dataset = {dataset.name}')
    print(modsel_choice)
    print(dataset_results)

plot_bandwidth(val_choice, test_results)





