import numpy as np


leaky_slope = 0.05


def len_check(y_pred,y_true):
    prediction_dim = y_pred.shape
    true_dim = y_true.shape
    if (prediction_dim!=true_dim):
        print(f'length of y_pred is {prediction_dim} while length of y_true is {true_dim}, there is a length mismatch. Raising error!')
        raise RuntimeError
    else:
        return


def cross_entropy_loss(y_pred, y_true):
    row_wiseSummed = np.sum(y_true*np.log(y_pred))
    loss = -np.mean(row_wiseSummed)
    return loss


def mean_squared_loss(y_pred, y_true)->np.ndarray:
    #1/N * summation (y_hat - y)^2
    loss = np.mean((y_pred-y_true)**2)
    return loss


def accuracy(y_pred, y_true)->np.ndarray:
    correct_preds = np.sum((y_pred==y_true),axis=1)
    total_preds = len(y_pred)
    accuracy = (correct_preds/total_preds)*100
    return accuracy


def softmax(x)->np.ndarray:
    x = x-np.max(x,axis=1,keepdims=True)
    sum_ofExps = np.sum(np.exp(x),axis=1,keepdims=True)
    softmax_vals = np.exp(x)/sum_ofExps
    return softmax_vals


def sigmoid(x)->np.ndarray:
    #clipping results for numerical stability
    x=np.clip(x,-500,500)
    activations = 1/(1+np.exp(-x))
    return activations


def Tanh(x)->np.ndarray:
    #clipping results for numerical stability
    x = np.clip(x,-500,500)
    activations = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return activations


def ReLU(x)->np.ndarray:
    #clipping results for numerical stability
    activations = np.where(x>0,x,0)
    return activations

def Leaky_ReLU(x)->np.ndarray:
    #clipping results for numerical stability
    activations = np.where(x>leaky_slope*x,x,leaky_slope*x)
    return activations

def test_len_check():
    print("Testing len_check...")
    y_pred = np.array([[0.1, 0.9], [0.8, 0.2]])
    y_true = np.array([[0, 1], [1, 0]])
    try:
        len_check(y_pred, y_true)
        print("  Pass: Shapes match")
        result = True
    except RuntimeError:
        print("  Fail: Shapes match but error raised")
        result = False

    y_true_mismatch = np.array([0, 1])
    try:
        len_check(y_pred, y_true_mismatch)
        print("  Fail: Shapes mismatch but no error raised")
        result = False
    except RuntimeError:
        print("  Pass: Shapes mismatch correctly raised error")
        result = result and True
    return result

def test_cross_entropy_loss():
    print("Testing cross_entropy_loss...")
    y_pred = np.array([[0.7, 0.3], [0.4, 0.6]])
    y_true = np.array([[1, 0], [0, 1]])
    loss = cross_entropy_loss(y_pred, y_true)
    print(f"  Loss: {loss}")
    return isinstance(loss, (float, np.floating))

def test_mean_squared_loss():
    print("Testing mean_squared_loss...")
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.0, 2.0, 2.0])
    loss = mean_squared_loss(y_pred, y_true)
    expected = np.mean((y_pred - y_true) ** 2)
    print(f"  Loss: {loss}")
    return np.isclose(loss, expected)

def test_accuracy():
    print("Testing accuracy...")
    y_pred = np.array([[1, 0, 1], [0, 1, 1]])
    y_true = np.array([[1, 0, 0], [0, 1, 1]])
    acc = accuracy(y_pred, y_true)
    print(f"  Accuracy per sample: {acc}")
    # Just check shape and type here
    return (acc.shape == (2,)) and (acc.dtype == float or acc.dtype == np.float64)

def test_softmax():
    print("Testing softmax...")
    x = np.array([[1, 2, 3], [3, 2, 1]])
    sm = softmax(x)
    print(f"  Softmax output:\n{sm}")
    row_sums = np.sum(sm, axis=1)
    print(f"  Row sums (should be 1): {row_sums}")
    return np.allclose(row_sums, 1)

def test_sigmoid():
    print("Testing sigmoid...")
    x = np.array([-1000, 0, 1000])
    s = sigmoid(x)
    print(f"  Sigmoid output: {s}")
    return np.all((s >= 0) & (s <= 1))

def test_tanh():
    print("Testing Tanh...")
    x = np.array([-1000, 0, 1000])
    t = Tanh(x)
    print(f"  Tanh output: {t}")
    return np.all((t >= -1) & (t <= 1))

def test_relu():
    print("Testing ReLU...")
    x = np.array([-2, 0, 3])
    r = ReLU(x)
    print(f"  ReLU output: {r}")
    return np.all(r >= 0)

def test_leaky_relu():
    print("Testing Leaky ReLU...")
    x = np.array([-2, 0, 3])
    l = Leaky_ReLU(x)
    print(f"  Leaky ReLU output: {l}")
    # Check output shape and some expected behavior
    cond1 = np.allclose(l[x > 0], x[x > 0])
    cond2 = np.all(l[x < 0] == leaky_slope * x[x < 0])
    return cond1 and cond2

if __name__ == "__main__":
    results = {
        "len_check": test_len_check(),
        "cross_entropy_loss": test_cross_entropy_loss(),
        "mean_squared_loss": test_mean_squared_loss(),
        "accuracy": test_accuracy(),
        "softmax": test_softmax(),
        "sigmoid": test_sigmoid(),
        "tanh": test_tanh(),
        "ReLU": test_relu(),
        "Leaky_ReLU": test_leaky_relu()
    }
    print("\nSummary of tests:")
    for func, passed in results.items():
        print(f"  {func}: {'PASS' if passed else 'FAIL'}")