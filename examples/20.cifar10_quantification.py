import quapy as qp
from quapy.data.datasets import fetch_image_embeddings
from quapy.method.aggregative import EMQ, RLLS
from quapy.classification.calibration import TemperatureScalingFromLogits
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression


# This example illustrates how to run experiments with image datasets, in this case with CIFAR10
# The datasets available in quapy do not consist of raw image files, but are instead pre-generated 
# embeddings (see the manuals for further information).

if __name__ == '__main__':

    # Let us begin with a typical case in which the embeddings come from the penultimate layer of a neural
    # model (in this case, a resnet18). We get these representations by specifying embedding='features'

    print('fetching cifar10 embeddings')
    train, test = fetch_image_embeddings(dataset_name='cifar10', embedding='features').train_test

    print('training:', train)
    print('test:', test)

    Xtr, ytr = train.Xy

    # let us train an Expectation Maximazion Quantifier (EMQ), aka Maximum Likelihood for Label Shift (MLLS)
    # using a logistic regressor as the underlying classifier, with Bias Corrected Temperature Scaling (BCTS)

    bcts_emq = EMQ(classifier=LogisticRegression(), calib='bcts', val_split=5)

    print(f'fitting quantifier {bcts_emq}')
    bcts_emq.fit(Xtr, ytr)

    # we generate many samples exhibiting prior probability shift with the artificial prevalence protocol
    # (we use the multiclass variant UPP instead of the grid-based APP)

    qp.environ["SAMPLE_SIZE"] = 500  # when the sample size is common to all experiments, it is conveniet to set it once and for all
    artificial_prev_prot = UPP(test, repeats=200)
    print('generating 200 test bags of 500 instances each')

    bctsemq_report = qp.evaluation.evaluation_report(bcts_emq, protocol=artificial_prev_prot, error_metrics=['mae', 'mrae'])
    print(bctsemq_report.mean(numeric_only=True))

    # we could instead use the pre-generated logits of the resnet18
    print('fetching cifar10 logits')
    train, test = fetch_image_embeddings(dataset_name='cifar10', embedding='logits').train_test
    Xtr, ytr = train.Xy

    # in this case, the representations are already classification-related outputs;
    # we can convert them into (hopefully well-) calibrated outputs via TemperatureScaling
    
    print('generating posterior probabilities out of logits via temperature scaling')    
    emq = EMQ(classifier=TemperatureScalingFromLogits(bias_corrected=True))
    emq.fit(Xtr, ytr)

    print('generating 200 test bags of 500 instances each')
    artificial_prev_prot = UPP(test, repeats=200)
    emq_report = qp.evaluation.evaluation_report(emq, protocol=artificial_prev_prot, error_metrics=['mae', 'mrae'])
    print(emq_report.mean(numeric_only=True))





