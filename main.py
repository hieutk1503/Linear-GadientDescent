import numpy as np
import pandas as pd
from GD import GD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

#đọc dữ liệu
def doc_du_lieu():
    file = os.path.join(os.path.dirname(__file__), "Student_Performance.csv")
    data=pd.read_csv(file)
    return data

def xu_li_du_lieu(data):
    X = data.iloc[:, :-1].drop(data.columns[2],axis=1).values  # Tất cả các cột trừ cột cuối và bỏ cột 3
    Y = data.iloc[:,-1].values.reshape(-1, 1)  #Lấy cột cuối
    return X,Y

def cap_nhat_Xbar(X):
    one=np.ones((X.shape[0],1))#tạo ma trận 1
    scaler = StandardScaler()# Chuẩn hóa X (dữ liệu đầu vào) bằng StandardScaler
    X_scaled =scaler.fit_transform(X)

    Xbar = np.concatenate((one, X_scaled), axis=1)# Cập nhật X với X_scaled đã chuẩn hóa
    return Xbar
if __name__=="__main__":
    data = doc_du_lieu()# Đọc dữ liệu
    X, Y = xu_li_du_lieu(data)
    # Chia dữ liệu thành tập huấn luyện (70%) và tập kiểm tra (30%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    Xbar_train = cap_nhat_Xbar(X_train) # Chuẩn hóa và thêm bias cho tập huấn luyện
    w = np.zeros((Xbar_train.shape[1], 1))  # Khởi tạo w với giá trị 0
    gd = GD(eta=0.001, Xbar=Xbar_train, Y=Y_train, w=w)# Khởi tạo và chạy gradient descent
    w_new, i = gd.run(max_iter=10000)  # Tìm trọng số và số lần lặp
    print('Trọng số w:', w_new)
    print('Vòng lặp kết thúc sau:', (i + 1), 'lần')
    Xbar_test = cap_nhat_Xbar(X_test)# Chuẩn hóa và thêm bias cho tập kiểm tra
    Y_dudoan = np.dot(Xbar_test, w_new)# Dự đoán với tập kiểm tra
    # In kết quả
    print("Giá trị thực tế Y:", Y_test.T)
    print("Giá trị dự đoán Y:", Y_dudoan.T)
    r2 = r2_score(Y_test, Y_dudoan)# Đánh giá độ chính xác
    print("Độ chính xác: ", r2)


