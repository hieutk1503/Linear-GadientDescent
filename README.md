# Tìm hiểu về Gradient Descent ứng dụng cho bài toán Linear Regression
## I. Giới thiệu chung.
### 1. Mô tả
Gradient Descent là một thuật toán tối ưu hóa được sử dụng rộng rãi trong học máy và thống kê để tìm cực tiểu của một hàm mất mát (loss function). Nó giúp điều chỉnh các tham số của mô hình sao cho sai số giữa giá trị dự đoán và giá trị thực tế nhỏ nhất.Dự án này nhằm tìm hiểu và triển khai thuật toán Gradient Descent trong bài toán hồi quy tuyến tính (Linear Regression). Dự án bao gồm lý thuyết cơ bản, cách áp dụng.
![Gradient-Descent-Top2](https://github.com/user-attachments/assets/e642c824-12fd-4b7c-82ae-e9f438783ee4)
### 2. Nội Dung Dự Án
Giới thiệu về Gradient Descent
Công thức toán học trong Linear Regression
Triển khai thuật toán bằng Python
Kết luận
## II. Giới thiệu về thuật toán Linear Regression.
### 1. Khái niệm
Linear Regression là một thuật toán học máy thuộc nhóm học có giám sát (Supervised Learning), được sử dụng để dự đoán các giá trị liên tục. Mô hình này thiết lập mối quan hệ tuyến tính giữa biến đầu vào (độc lập) và biến đầu ra (phụ thuộc).
### 2. Ý tưởng chính
Linear Regression giả định rằng có một mối quan hệ tuyến tính giữa biến đầu vào và biến đầu ra:

![CT1_Linear](https://github.com/user-attachments/assets/ffe339e8-0d24-438c-9217-caf776662995)

Dưới dạng vecto:

![Screenshot 2024-12-20 013845](https://github.com/user-attachments/assets/5af95bd8-e582-4857-af8f-587bbe7c220b)
### 3.Hàm mất mát

![Screenshot 2024-12-20 014004](https://github.com/user-attachments/assets/8e77b981-8feb-4759-b94e-66ec92c33f84)
Mục tiêu là làm cho hàm mất mát nhỏ nhất

## III. Giới thiệu về Gradient Descent.
### 1. Khái niệm
Gradient Descent là một thuật toán tối ưu hóa được sử dụng để tìm giá trị cực tiểu của một hàm số, thường áp dụng trong học máy và học sâu. Mục tiêu là điều chỉnh các tham số mô hình để giảm thiểu hàm mất mát (loss function).
### 2. Công thức
![image](https://github.com/user-attachments/assets/0b0456c0-4a39-4a5b-a625-3bcc860e6868)

Cách hoạt động: Gradient Descent dựa trên việc cập nhật các trọng số theo hướng ngược lại với đạo hàm của hàm mất mát và thuật toán sẽ dừng khi ∇L(w) là 1 số gần bằng 0, thường là 1e-5.
## IV. Cài đặt
- Lớp main kế thừa lớp GD để chạy chương trình chính.
- Sử dụng các thư viện Pandas để đọc dữ liệu, thư viện sklearn để chuẩn hoá dữ liệu và tính độ chính xác. 
