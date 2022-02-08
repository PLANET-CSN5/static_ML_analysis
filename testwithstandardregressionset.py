import sys
import smlmodule
import inflect
from sklearn.datasets import make_regression

N = 10
X, y = make_regression(n_samples=1000, n_features=N, n_informative=5, random_state=1)
print("Test the procedure")
print("X shape: ", X.shape, " and y shape: ", y.shape)

p = inflect.engine()

features = []
for i in range(N):
    features.append(p.number_to_words(i+1))

smlmodule.rfregressors (X, y, features, plotname="rf_model", N = 50, verbose=True,
    pout=sys.stdout)