# K-nearest neighbors
# Thuật toán này áp dụng cho cả 2 loại thuật toán phân loại và hồi quy
# Trong bài toán phân loại thì đầu ra là nhãn của điểm gần nó nhất(các điểm gần nó nhất)
# trong bài toán hòi quy thì đầu ra của nó là giá trij gần nó nhất hoặc giá trị trung bình gần nó nhất
# Nó không có hàm mất mát và baif toán tối ưu nào
# Mọi tính toán được thực hiện ở kiểm thử
# Mối điểm dữ liệu là 1 vector đặc trưng khoảng cach giữa 2 điểm là khoảng cách giữa 2 vector đó, và dùng norm l2(chính là Eclid )


# Dưới là demo cách tính khoagr cách. Nếu tự viết thuật toán thì sau khi tính dc khảng cách rồi thì lưu nó vào và sắp
# xếp rồi lấy K phần tử (chính là số hàng xóm)
# rồi áp dụng vào từng bài toán xắp xếp hay hồi quy

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from time import time

N = 10000
d = 1000

X = np.random.randn(N, d)
z = np.random.randn(d)
z2 = np.sum(z * z)


# print(z2)


# tính khoảng cách giữa 2 vector

def dist_pp(z, x):
    d = z - x.reshape(z.shape)  # x và z phải có cùng dim chiều nếu là vector doc thì cùng là vector doc
    return np.sum(d * d)


# Tính khoảng cách từ 1 điểm đến 1 điểm trong tập hợp X (cụ thể ở đây là z tới X[i])
def dist_ps_naive(z, X):
    N = X.shape[0]  # lấy ra kích thước của ma trận cột đấy
    res = np.zeros((1, N))  # tạo ra 1 ma trận hàng N phần tử
    for i in range(N):
        res[0][i] = dist_pp(z, X[i])
    return res


# cách này nhanh hơn
def dist_ps_fast(z, X):
    X2 = np.sum(X * X, 1)
    z2 = np.sum(z * z)
    return X2 + z2 - 2 * X.dot(z)


t1 = time()
D1 = dist_ps_naive(z, X)
print('naive :', time() - t1, 's')

t1 = time()
D2 = dist_ps_fast(z, X)

print('fast :', time() - t1, 's')
print('Ket qua khac biet  : ', np.linalg.norm(D1 - D2))
