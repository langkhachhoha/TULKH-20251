# Vehicle Routing Problem with Time Windows (VRPTW) Solver

Bộ giải thuật tối ưu cho bài toán **Vehicle Routing Problem with Time Windows** sử dụng 2 thuật toán metaheuristic với tổng cộng 12 biến thể.

## 📋 Mô tả bài toán

Một nhân viên giao hàng xuất phát từ kho (điểm 0) và cần giao hàng cho N khách hàng (1, 2, ..., N). Mỗi khách hàng i có:
- **Time window**: Phải được giao hàng trong khoảng thời gian từ `e(i)` đến `l(i)`
- **Service duration**: Giao hàng mất `d(i)` đơn vị thời gian
- **Travel time**: Thời gian di chuyển từ điểm i đến j là `t(i,j)`

**Mục tiêu**: Tìm lộ trình giao hàng sao cho tổng thời gian di chuyển là **ngắn nhất** và thỏa mãn tất cả ràng buộc time window.

## 📁 Cấu trúc dự án

```
TULKH/
├── local_search.py      # Local Search với 6 biến thể
├── tabu_search.py       # Tabu Search với 6 biến thể
└── README.md           # File hướng dẫn này
```

## 🔍 Các thuật toán

### 1. Local Search (`local_search.py`)

Thuật toán tìm kiếm cục bộ với 6 biến thể khác nhau:

| Mode | Tên thuật toán | Mô tả |
|------|----------------|-------|
| **1** | 2-opt with restart | 2-opt với restart mechanism và first improvement |
| **2** | Variable Neighborhood Descent | Kết hợp 3 loại move: swap, insert, reverse |
| **3** | Iterated Local Search | ILS với perturbation có độ mạnh thay đổi |
| **4** | Simulated Annealing | SA với temperature cooling và multiple moves |
| **5** | GRASP | Greedy Randomized Adaptive Search Procedure |
| **6** | Late Acceptance Hill Climbing | Hill climbing với late acceptance criterion |

**Đặc điểm:**
- ✅ Nhanh, phù hợp với bài toán nhỏ-trung bình
- ✅ 3 chiến lược khởi tạo khác nhau
- ✅ First improvement để tăng tốc
- ✅ Escape mechanisms để thoát local optima

### 2. Tabu Search (`tabu_search.py`)

Thuật toán Tabu Search với 6 biến thể nâng cao:

| Mode | Tên thuật toán | Mô tả |
|------|----------------|-------|
| **1** | Reactive Tabu Search | Adaptive tabu tenure với intensification/diversification |
| **2** | Robust Tabu Search | Strategic oscillation với multiple move types |
| **3** | Adaptive Tabu Search | Aspiration plus với adaptive parameters |
| **4** | Path Relinking Tabu Search | Elite solutions pool với path relinking |
| **5** | Granular Tabu Search | Candidate lists dựa trên cấu trúc bài toán |
| **6** | Probabilistic Tabu Search | Probabilistic tabu với threshold accepting |

**Đặc điểm:**
- ✅ Chất lượng cao, phù hợp với bài toán trung bình-lớn
- ✅ Memory structures (tabu list, frequency, elite pool)
- ✅ Multiple move types và aspiration criteria
- ✅ Diversification mechanisms

## 🚀 Cách sử dụng

### 1. Cài đặt

Yêu cầu: **Python 3.8+**

Không cần cài đặt thư viện bổ sung (chỉ dùng thư viện chuẩn).

### 2. Chạy thuật toán

#### Bước 1: Chọn MODE

Mở file thuật toán và chỉnh biến `MODE`:

```python
# Trong local_search.py hoặc tabu_search.py
MODE = 1  # Đổi thành 1, 2, 3, 4, 5, hoặc 6
```

#### Bước 2: Chạy với input

**Từ file:**
```bash
python local_search.py < input.txt
python tabu_search.py < input.txt
```

**Từ stdin:**
```bash
python local_search.py
# Nhập dữ liệu theo format bên dưới
```

### 3. Format Input

```
N
e(1) l(1) d(1)
e(2) l(2) d(2)
...
e(N) l(N) d(N)
t(0,0) t(0,1) ... t(0,N)
t(1,0) t(1,1) ... t(1,N)
...
t(N,0) t(N,1) ... t(N,N)
```

**Trong đó:**
- `N`: Số lượng khách hàng (1 ≤ N ≤ 1000)
- `e(i), l(i), d(i)`: Earliest time, latest time, duration của khách hàng i
- `t(i,j)`: Ma trận thời gian di chuyển (N+1) × (N+1)

### 4. Format Output

```
N
s(1) s(2) ... s(N)
```

**Trong đó:**
- `N`: Số lượng khách hàng
- `s(1) s(2) ... s(N)`: Thứ tự giao hàng (permutation của 1..N)

## 📊 Ví dụ

### Input (`input.txt`)

```
5
50 90 20
300 350 15
215 235 5
374 404 20
107 147 20
0 50 10 100 70 10
50 0 40 70 20 40
10 40 0 80 60 0
100 70 80 0 70 80
70 20 60 70 0 60
10 40 0 80 60 0
```

### Chạy thuật toán

```bash
# Local Search Mode 3 (ILS)
python local_search.py < input.txt

# Tabu Search Mode 1 (Reactive)
python tabu_search.py < input.txt
```

### Output

```
5
1 5 3 2 4
```

## ⚙️ Cấu hình tham số

### Local Search

Các tham số có thể điều chỉnh trong code:

```python
# Số iterations tối đa
max_iter = 10000

# Cho GRASP (Mode 5)
num_constructions = 10
alpha = 0.3  # Randomization parameter

# Cho SA (Mode 4)
temp = 100.0
cooling_rate = 0.995
min_temp = 0.1
```

### Tabu Search

```python
# Số iterations tối đa
max_iter = 2000

# Tabu tenure
tabu_tenure = 7-15  # Tùy mode

# Path Relinking (Mode 4)
max_elite = 5

# Granular (Mode 5)
granular_threshold = 100  # Time window threshold
```

## 📈 So sánh thuật toán

| Tiêu chí | Local Search | Tabu Search |
|----------|--------------|-------------|
| **Tốc độ** | ⭐⭐⭐⭐⭐ Rất nhanh | ⭐⭐⭐⭐ Nhanh |
| **Chất lượng** | ⭐⭐⭐ Tốt | ⭐⭐⭐⭐⭐ Rất tốt |
| **Bài toán nhỏ (N<50)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Bài toán lớn (N>100)** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Đơn giản** | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 💡 Khuyến nghị

### Chọn thuật toán

- **Local Search**: Dùng khi cần kết quả nhanh, bài toán nhỏ-trung bình
  - Mode 1-3: Nhanh nhất, chất lượng tốt
  - Mode 4-5: Cân bằng tốc độ và chất lượng
  - Mode 6: Thử nghiệm khi các mode khác stuck

- **Tabu Search**: Dùng khi cần chất lượng cao, bài toán trung bình-lớn
  - Mode 1: Tốt nhất cho hầu hết trường hợp
  - Mode 3-4: Chất lượng cao nhất
  - Mode 5: Tốt cho bài toán có cấu trúc đặc biệt
  - Mode 6: Alternative approach khi cần exploration

### Tips để có kết quả tốt

1. **Thử nhiều modes**: Mỗi mode phù hợp với structure khác nhau
2. **Chạy nhiều lần**: Các thuật toán có yếu tố random
3. **Điều chỉnh parameters**: Tùy theo kích thước bài toán
4. **Kết hợp**: Dùng Local Search để khởi tạo, sau đó Tabu Search để refine

## 🔧 Cải tiến đã implement

### Khởi tạo thông minh
- ✅ Nearest neighbor heuristic
- ✅ Time window-based sorting (earliest, latest)
- ✅ Multiple initial solutions

### Local improvement
- ✅ 2-opt moves
- ✅ Insert moves
- ✅ Reverse segment
- ✅ First improvement strategy

### Escape mechanisms
- ✅ Restart with perturbation
- ✅ Simulated Annealing acceptance
- ✅ Late acceptance
- ✅ Threshold accepting

### Memory structures
- ✅ Tabu list
- ✅ Frequency-based memory
- ✅ Elite solutions pool
- ✅ Aspiration criteria

### Advanced techniques
- ✅ Variable neighborhood search
- ✅ Path relinking
- ✅ Granular search
- ✅ Strategic oscillation
- ✅ Adaptive parameters

## 📝 Giải thích thuật toán

### Local Search

**Ý tưởng chính**: Bắt đầu từ một solution, liên tục cải thiện bằng cách thử các moves (swap, insert, reverse) cho đến khi không cải thiện được nữa.

**Cách hoạt động**:
1. Generate initial solution
2. Explore neighborhood (các solutions gần)
3. Move to better solution
4. Repeat until no improvement
5. Apply escape mechanism if stuck

### Tabu Search

**Ý tưởng chính**: Giống Local Search nhưng có "bộ nhớ" (tabu list) để tránh quay lại các solutions đã thăm gần đây, cho phép escape khỏi local optima.

**Cách hoạt động**:
1. Generate initial solution
2. Explore neighborhood
3. Move to best non-tabu neighbor (kể cả worse)
4. Add move to tabu list
5. Update best solution if found
6. Repeat for max iterations
7. Apply advanced mechanisms (aspiration, diversification, etc.)

## 🐛 Xử lý lỗi

### Infeasible solutions

Thuật toán tự động xử lý:
- Solutions vi phạm time windows → cost = infinity
- Không tìm được feasible neighbor → restart/perturbation

### Time windows quá chặt

Nếu không tìm được solution khả thi:
- Kiểm tra lại input (travel time, time windows)
- Giảm strictness của time windows
- Tăng số iterations

## 📚 Tài liệu tham khảo

### Papers
- **Tabu Search**: Glover, F. (1989). "Tabu Search—Part I"
- **ILS**: Lourenço, H. R., et al. (2003). "Iterated Local Search"
- **GRASP**: Feo, T. A., & Resende, M. G. (1995). "Greedy Randomized Adaptive Search Procedures"
- **VRPTW**: Solomon, M. M. (1987). "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints"

### Books
- "Handbook of Metaheuristics" - Gendreau & Potvin (2019)
- "Local Search in Combinatorial Optimization" - Aarts & Lenstra (1997)

## 👨‍💻 Tác giả

Được phát triển cho môn Tối ưu hóa / Metaheuristics.

## 📄 License

Free to use for educational purposes.

---

**Chúc bạn tối ưu thành công! 🚀**

Nếu có vấn đề hoặc câu hỏi, hãy thử các modes khác nhau hoặc điều chỉnh parameters.

