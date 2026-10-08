import numpy as np
from GD import GD

class LR(GD):
    def __init__(self, eta=0.001, Xbar=None, Y=None, w=None, X=None):
        if Xbar is None and X is not None:
            Xbar = X
        super().__init__(eta, Xbar, Y, w)
        self.X = Xbar

    # hàm mất mát
    def tinh_cost(self, Xbar=None, Y=None, w=None):
        if Xbar is None:
            Xbar = self.Xbar if self.Xbar is not None else self.X
        if Y is None:
            Y = self.Y
        if w is None:
            w = self.w
        if Xbar is None or Y is None or w is None:
            return 0.0
        m = len(Y) if len(Y) > 0 else 1
        predictions = np.dot(Xbar, w)  # giá trị của Y dự đoán
        cost = (1 / (2 * m)) * np.sum((Y - predictions) ** 2)  # hàm mất mát
        return cost

    # đạo hàm hàm mất mát
    def tinh_grad(self, Xbar=None, Y=None, w=None):
        if Xbar is None:
            Xbar = self.Xbar if self.Xbar is not None else self.X
        if Y is None:
            Y = self.Y
        if w is None:
            w = self.w
        m = len(Y) if len(Y) > 0 else 1
        predictions = np.dot(Xbar, w)  # giá trị của Y dự đoán
        gradient = (1 / m) * np.dot(Xbar.T, (predictions - Y))  # giá trị đạo hàm hàm mất mát
        return gradient

    # tính nghiệm bằng Gradient Descent
    def run(self, max_iter=10000, *args, **kwargs):
        # Hỗ trợ cả 2 cách gọi: run(max_iter) hoặc run(Xbar, Y, w, max_iter=...)
        if isinstance(max_iter, (np.ndarray, list)):
            Xbar = max_iter
            Y = args[0] if len(args) > 0 else self.Y
            w = args[1] if len(args) > 1 else self.w
            iter_count = args[2] if len(args) > 2 else kwargs.get('max_iter', 10000)
            self.Xbar = Xbar
            self.X = Xbar
            self.Y = Y
            self.w = np.copy(w)
            max_iter = iter_count

        m = self.Xbar.shape[0]  # số dữ liệu
        i = 0
        for i in range(max_iter):  # Vòng lặp
            gradient = self.tinh_grad()  # giá trị đạo hàm
            self.w = self.w - (self.eta * gradient)  # công thức của gradient descent
            if np.linalg.norm(gradient) / m < 1e-5:  # thuật toán kết thúc khi giá trị đạo hàm nhỏ hơn 1e-5
                break
        return self.w, i

    # tính nghiệm theo công thức giải tích (Normal Equation): w = (Xbar^T * Xbar)^(-1) * Xbar^T * Y
    def fit_normal_equation(self, Xbar=None, Y=None):
        if Xbar is None:
            Xbar = self.Xbar if self.Xbar is not None else self.X
        if Y is None:
            Y = self.Y
        A = np.dot(Xbar.T, Xbar)
        b = np.dot(Xbar.T, Y)
        self.w = np.dot(np.linalg.pinv(A), b)
        return self.w

    # dự đoán giá trị Y
    def predict(self, Xbar=None):
        if Xbar is None:
            Xbar = self.Xbar if self.Xbar is not None else self.X
        return np.dot(Xbar, self.w)


# Định danh bổ sung tương thích
LinearRegression = LR

if __name__ == "__main__":
    import os
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    # Đọc dữ liệu
    file_path = "Student_Performance.csv"
    if not os.path.exists(file_path):
        file_path = "\\Python\\Code_Python\\Student_Performance.csv"

    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].drop(data.columns[2], axis=1).values
        Y = data.iloc[:, -1].values.reshape(-1, 1)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        def cap_nhat_Xbar(X_data):
            one = np.ones((X_data.shape[0], 1))
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)
            return np.concatenate((one, X_scaled), axis=1)

        Xbar_train = cap_nhat_Xbar(X_train)
        Xbar_test = cap_nhat_Xbar(X_test)

        w = np.zeros((Xbar_train.shape[1], 1))

        print("=== Test Linear Regression voi Gradient Descent ===")
        lr = LR(eta=0.001, Xbar=Xbar_train, Y=Y_train, w=w)
        w_new, i = lr.run(max_iter=10000)
        print("Trong so w (GD):", w_new.ravel())
        print("Vong lap ket thuc sau:", (i + 1), "lan")
        Y_dudoan = lr.predict(Xbar_test)
        print("Do chinh xac R2 (GD):", r2_score(Y_test, Y_dudoan))

        print("\n=== Test Linear Regression voi Normal Equation ===")
        lr_norm = LR(Xbar=Xbar_train, Y=Y_train)
        w_norm = lr_norm.fit_normal_equation()
        print("Trong so w (Normal Eq):", w_norm.ravel())
        Y_dudoan_norm = lr_norm.predict(Xbar_test)
        print("Do chinh xac R2 (Normal Eq):", r2_score(Y_test, Y_dudoan_norm))
