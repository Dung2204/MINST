import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
import struct
import altair
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from PIL import Image
from collections import Counter


@st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
def get_sampled_pixels(images, sample_size=100_000):
    return np.random.choice(images.flatten(), sample_size, replace=False)

@st.cache_data  # Cache danh sách ảnh ngẫu nhiên
def get_random_indices(num_images, total_images):
    return np.random.randint(0, total_images, size=num_images)

# Cấu hình Streamlit
st.set_page_config(page_title="Phân loại ảnh", layout="wide")
# Định nghĩa hàm để đọc file .idx
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Định nghĩa đường dẫn đến các file MNIST
dataset_path = r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\BaiThucHanh3"
train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

# Tải dữ liệu
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# Giao diện Streamlit
st.title("📸 Phân loại ảnh MNIST với Streamlit")

with st.expander("🖼️ Dữ liệu ban đầu", expanded=True):
    st.subheader("📌***1.Thông tin về bộ dữ liệu MNIST***")
    st.markdown(
        '''
        **MNIST** là phiên bản được chỉnh sửa từ bộ dữ liệu **NIST gốc** của Viện Tiêu chuẩn và Công nghệ Quốc gia Hoa Kỳ.  
        Bộ dữ liệu ban đầu gồm các chữ số viết tay từ **nhân viên bưu điện** và **học sinh trung học**.  

        Các nhà nghiên cứu **Yann LeCun, Corinna Cortes, và Christopher Burges** đã xử lý, chuẩn hóa và chuyển đổi bộ dữ liệu này thành **MNIST**  
        để dễ dàng sử dụng hơn cho các bài toán nhận dạng chữ số viết tay.
        '''
    )
    # Đặc điểm của bộ dữ liệu
    st.subheader("📌***2. Đặc điểm của bộ dữ liệu***")
    st.markdown(
        '''
        - **Số lượng ảnh:** 70.000 ảnh chữ số viết tay  
        - **Kích thước ảnh:** Mỗi ảnh có kích thước 28x28 pixel  
        - **Cường độ điểm ảnh:** Từ 0 (màu đen) đến 255 (màu trắng)  
        - **Dữ liệu nhãn:** Mỗi ảnh đi kèm với một nhãn số từ 0 đến 9  
        '''
    )
    st.write(f"🔍 Số lượng ảnh huấn luyện: `{train_images.shape[0]}`")
    st.write(f"🔍 Số lượng ảnh kiểm tra: `{test_images.shape[0]}`")


    st.subheader("📌***3. Hiển thị số lượng mẫu của từng chữ số từ 0 đến 9 trong tập huấn luyện**")
    label_counts = pd.Series(train_labels).value_counts().sort_index()
    # Hiển thị biểu đồ cột
    st.subheader("📊 Biểu đồ số lượng mẫu của từng chữ số")
    st.bar_chart(label_counts)
    # Hiển thị bảng dữ liệu dưới biểu đồ
    st.subheader("📋 Số lượng mẫu cho từng chữ số")
    df_counts = pd.DataFrame({"Chữ số": label_counts.index, "Số lượng mẫu": label_counts.values})
    st.dataframe(df_counts)


    st.subheader("📌***4. Chọn ngẫu nhiên 10 ảnh từ tập huấn luyện để hiển thị***")
    num_images = 10
    random_indices = random.sample(range(len(train_images)), num_images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for ax, idx in zip(axes, random_indices):
        ax.imshow(train_images[idx], cmap='gray')
        ax.axis("off")
        ax.set_title(f"Label: {train_labels[idx]}")

    st.pyplot(fig)

    st.subheader("📌***5. Kiểm tra hình dạng của tập dữ liệu***")
        # Kiểm tra hình dạng của tập dữ liệu
    st.write("🔍 Hình dạng tập huấn luyện:", train_images.shape)
    st.write("🔍 Hình dạng tập kiểm tra:", test_images.shape)

    st.subheader("📌***6. Kiểm tra xem có giá trị NaN và phù hợp trong phạm vi không***")
    # Kiểm tra xem có giá trị NaN không
    if np.isnan(train_images).any() or np.isnan(test_images).any():
        st.error("⚠️ Cảnh báo: Dữ liệu chứa giá trị NaN!")
    else:
        st.success("✅ Không có giá trị NaN trong dữ liệu.")

    # Kiểm tra xem có giá trị pixel nào ngoài phạm vi 0-255 không
    if (train_images.min() < 0) or (train_images.max() > 255):
        st.error("⚠️ Cảnh báo: Có giá trị pixel ngoài phạm vi 0-255!")
    else:
        st.success("✅ Dữ liệu pixel hợp lệ (0 - 255).")



    st.subheader("📌***7. Chuẩn hóa dữ liệu (đưa giá trị pixel về khoảng 0-1)***")
    # Chuẩn hóa dữ liệu
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Hiển thị thông báo sau khi chuẩn hóa
    st.success("✅ Dữ liệu đã được chuẩn hóa về khoảng [0,1].")

    # Hiển thị bảng dữ liệu đã chuẩn hóa (dạng số)
    num_samples = 5  # Số lượng mẫu hiển thị
    df_normalized = pd.DataFrame(train_images[:num_samples].reshape(num_samples, -1))  

    st.subheader("📌 **Bảng dữ liệu sau khi chuẩn hóa**")
    st.dataframe(df_normalized)

    
    sample_size = 10_000  
    pixel_sample = np.random.choice(train_images.flatten(), sample_size, replace=False)

    st.subheader("📊 **Phân bố giá trị pixel sau khi chuẩn hóa**")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Vẽ histogram tối ưu hơn
    ax.hist(pixel_sample, bins=30, color="blue", edgecolor="black")
    ax.set_title("Phân bố giá trị pixel sau khi chuẩn hóa", fontsize=12)
    ax.set_xlabel("Giá trị pixel (0-1)")
    ax.set_ylabel("Tần suất")

    st.pyplot(fig)
    st.markdown(
    """
    **🔍 Giải thích:**

        1️⃣ Phần lớn pixel có giá trị gần 0: 
        - Cột cao nhất nằm ở giá trị pixel ~ 0 cho thấy nhiều điểm ảnh trong tập dữ liệu có màu rất tối (đen).  
        - Điều này phổ biến trong các tập dữ liệu grayscale như **MNIST** hoặc **Fashion-MNIST**.  

        2️⃣ Một lượng nhỏ pixel có giá trị gần 1:
        - Một số điểm ảnh có giá trị pixel gần **1** (màu trắng), nhưng số lượng ít hơn nhiều so với pixel tối.  

        3️⃣ Rất ít pixel có giá trị trung bình (0.2 - 0.8):
        - Phân bố này cho thấy hình ảnh trong tập dữ liệu có độ tương phản cao.  
        - Phần lớn pixel là **đen** hoặc **trắng**, ít điểm ảnh có sắc độ trung bình (xám).  
    """
    )


# 🖼️ XỬ LÝ DỮ LIỆU
with st.expander("🖼️ XỬ LÝ DỮ LIỆU", expanded=True):
    st.header("📌 8. Xử lý dữ liệu và chuẩn bị huấn luyện")

    # Chuyển đổi dữ liệu thành vector 1 chiều
    X_train = train_images.reshape(train_images.shape[0], -1)
    X_test = test_images.reshape(test_images.shape[0], -1)

    # Chia tập train thành train/validation (80% - 20%)
    X_train, X_val, y_train, y_val = train_test_split(X_train, train_labels, test_size=0.2, random_state=42)

    st.write("✅ Dữ liệu đã được xử lý và chia tách.")
    st.write(f"🔹 Kích thước tập huấn luyện: `{X_train.shape}`")
    st.write(f"🔹 Kích thước tập validation: `{X_val.shape}`")
    st.write(f"🔹 Kích thước tập kiểm tra: `{X_test.shape}`")

    # Biểu đồ phân phối nhãn dữ liệu
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(Counter(y_train).keys()), y=list(Counter(y_train).values()), palette="Blues", ax=ax)
    ax.set_title("Phân phối nhãn trong tập huấn luyện")
    ax.set_xlabel("Nhãn")
    ax.set_ylabel("Số lượng")
    st.pyplot(fig)
    st.markdown(
    """
    ### 📊 Mô tả biểu đồ  
    Biểu đồ cột hiển thị **phân phối nhãn** trong tập huấn luyện.  
    - **Trục hoành (x-axis):** Biểu diễn các nhãn (labels) từ `0` đến `9`.  
    - **Trục tung (y-axis):** Thể hiện **số lượng mẫu dữ liệu** tương ứng với mỗi nhãn.  

    ### 🔍 Giải thích  
    - Biểu đồ giúp ta quan sát số lượng mẫu của từng nhãn trong tập huấn luyện.  
    - Mỗi thanh (cột) có màu sắc khác nhau: **xanh nhạt đến xanh đậm**, đại diện cho số lượng dữ liệu của từng nhãn.  
    - Một số nhãn có số lượng mẫu nhiều hơn hoặc ít hơn, điều này có thể gây ảnh hưởng đến độ chính xác của mô hình nếu dữ liệu không cân bằng.  
  
    """
    )




# 2️⃣ HUẤN LUYỆN CÁC MÔ HÌNH
with st.expander("📌 HUẤN LUYỆN MÔ HÌNH", expanded=True):
    st.header("📌 9. Huấn luyện các mô hình phân loại")

    # Decision Tree Classifier
    st.subheader("🌳 Decision Tree Classifier")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_val_pred_dt = dt_model.predict(X_val)
    accuracy_dt = accuracy_score(y_val, y_val_pred_dt)
    st.write(f"✅ **Độ chính xác trên tập validation:** `{accuracy_dt:.4f}`")

    # SVM Classifier
    st.subheader("🌀 Support Vector Machine (SVM)")
    svm_model = SVC(kernel="linear", random_state=42)
    svm_model.fit(X_train, y_train)
    y_val_pred_svm = svm_model.predict(X_val)
    accuracy_svm = accuracy_score(y_val, y_val_pred_svm)
    st.write(f"✅ **Độ chính xác trên tập validation:** `{accuracy_svm:.4f}`")

    # So sánh độ chính xác giữa Decision Tree và SVM
    fig, ax = plt.subplots(figsize=(6, 4))
    models = ["Decision Tree", "SVM"]
    accuracies = [accuracy_dt, accuracy_svm]
    sns.barplot(x=models, y=accuracies, palette="coolwarm", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("So sánh độ chính xác trên tập validation")
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)
    st.markdown(
    """
    ### 🔍 Nhận xét:
    - ✅ **Mô hình SVM có độ chính xác cao hơn** so với Decision Tree.
    - 📊 **Decision Tree**: Độ chính xác khoảng **85%**.
    - 📈 **SVM**: Độ chính xác gần **95%**.
    - 🚀 Điều này cho thấy **SVM hoạt động tốt hơn** trên tập validation, có thể do khả năng tổng quát hóa tốt hơn so với Decision Tree.
    """
    )



# 3️⃣ ĐÁNH GIÁ MÔ HÌNH
with st.expander("📌 ĐÁNH GIÁ MÔ HÌNH", expanded=True):
    st.header("📌 10. Đánh giá mô hình bằng Confusion Matrix")

    # Chọn mô hình tốt nhất
    best_model = dt_model if accuracy_dt > accuracy_svm else svm_model
    best_model_name = "Decision Tree" if accuracy_dt > accuracy_svm else "SVM"

    st.write(f"🏆 **Mô hình có độ chính xác cao nhất trên validation:** `{best_model_name}`")

    # Dự đoán trên tập kiểm tra
    y_test_pred = best_model.predict(X_test)

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(test_labels, y_test_pred, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix trên tập kiểm tra")
    st.pyplot(fig)
    st.markdown(
    """
    ### 📌 Giải thích hình ảnh:
    - **Trục hoành (x-axis):** Nhãn dự đoán của mô hình (**Predicted label**).
    - **Trục tung (y-axis):** Nhãn thực tế (**True label**).
    - **Các ô màu xanh đậm:** Số lượng mẫu được mô hình phân loại đúng.
    - **Các ô nhạt hơn:** Số lượng mẫu bị dự đoán sai thành một nhãn khác.
    
    ### 🔍 Ý nghĩa quan trọng:
    - **Các số trên đường chéo chính** (từ trên trái xuống dưới phải) đại diện cho số lượng dự đoán đúng của mô hình cho từng nhãn.
    - **Các số ngoài đường chéo** là những mẫu bị phân loại sai.
    - **Ô màu xanh đậm nhất** (ví dụ: ô `[1,1]` với giá trị `1119`) có nghĩa là mô hình dự đoán đúng `1119` mẫu thuộc lớp `1`.
    - **Ô `[5,3]` có giá trị `38`** → Mô hình nhầm `38` mẫu của lớp `5` thành lớp `3`.
    """
)

     
# 4️⃣ DEMO DỰ ĐOÁN TRÊN 10 ẢNH KIỂM TRA
with st.expander("📌 DEMO DỰ ĐOÁN", expanded=True):
    st.header("📌 11. Demo dự đoán trên ảnh kiểm tra")

    num_demo_images = 10
    random_indices = np.random.choice(len(X_test), num_demo_images, replace=False)

    fig, axes = plt.subplots(1, num_demo_images, figsize=(15, 5))

    for ax, idx in zip(axes, random_indices):
        ax.imshow(test_images[idx], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Dự đoán: {best_model.predict(X_test[idx].reshape(1, -1))[0]}")

    st.pyplot(fig)








# # # cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\BaiThucHanh3"
