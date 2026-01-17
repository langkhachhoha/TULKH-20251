# VRPTW Optimization Project

Dự án tối ưu hóa bài toán Vehicle Routing Problem with Time Windows (VRPTW) sử dụng nhiều thuật toán metaheuristic khác nhau.

## Cấu trúc dự án

```
TULKH/
├── main.py              # File chính để chạy tất cả các thuật toán
├── local_search.py      # 6 modes của Local Search
├── tabu_search.py       # 6 modes của Tabu Search (chỉ chạy 3 mode đầu)
├── GA.py                # Genetic Algorithm
├── result.txt           # Kết quả chạy thuật toán
└── README.md            # File này

Input folder:
/Users/apple/Downloads/TSP_timeWindow_OptimalPlanning_20222_HUST-main/testcase/input/
├── script.py            # Script tạo test cases tự động
├── N5.txt, N10.txt, ... # Test cases gốc (10 files)
└── N5_v1.txt, ...       # Test cases sinh tự động (50 files)
```

## Các thuật toán

### Exact Algorithms (Thuật toán chính xác)

#### 1. Backtracking
- Thuật toán quay lui với pruning
- Tìm nghiệm tối ưu chính xác
- Chỉ áp dụng cho bài toán nhỏ (N ≤ 10)
- Độ phức tạp: O(N!)

#### 2. Branch and Bound
- Thuật toán nhánh cận với priority queue
- Sử dụng lower bound để cắt tỉa
- Hiệu quả hơn backtracking
- Chỉ áp dụng cho bài toán nhỏ (N ≤ 15)

#### 3. OR-Tools
- Sử dụng thư viện Google OR-Tools
- Constraint Programming solver
- Yêu cầu cài đặt: `pip install ortools`
- Có thể giải bài toán lớn hơn

### Metaheuristic Algorithms (Thuật toán xấp xỉ)

### Local Search (6 modes)
1. **Mode 1**: 2-opt with restart and first improvement
2. **Mode 2**: Variable neighborhood descent (2-opt + insert + swap)
3. **Mode 3**: Iterated local search with perturbation
4. **Mode 4**: Simulated annealing with 2-opt
5. **Mode 5**: GRASP (Greedy Randomized Adaptive Search)
6. **Mode 6**: Late acceptance hill climbing

### Tabu Search (3 modes đầu)
1. **Mode 1**: Reactive tabu search with intensification/diversification
2. **Mode 2**: Robust tabu search with strategic oscillation
3. **Mode 3**: Adaptive tabu search with aspiration plus

### Genetic Algorithm
- Population-based evolutionary algorithm
- Order Crossover (OX)
- Swap and Inversion mutations
- Elitism strategy
- Local search improvement (2-opt)

## Giới hạn thời gian

Mỗi phương pháp có giới hạn thời gian chạy dựa trên kích thước bài toán:

| Kích thước (N) | Thời gian tối đa |
|----------------|------------------|
| N ≤ 100        | 60 giây          |
| 100 < N ≤ 500  | 120 giây         |
| N > 500        | 180 giây         |

## Test Cases

### Test cases gốc (10 files)
- N5.txt, N10.txt, N100.txt, N200.txt, N300.txt
- N500.txt, N600.txt, N700.txt, N900.txt, N1000.txt

### Test cases tự động sinh (50 files)
- 5 phiên bản cho mỗi kích thước: N5_v1.txt đến N5_v5.txt, ...
- Tổng cộng: **60 test cases**

## Cách sử dụng

### 1. Cài đặt (Optional - cho OR-Tools)

```bash
pip install ortools
```

### 2. Tạo test cases mới

```bash
cd /Users/apple/Downloads/TSP_timeWindow_OptimalPlanning_20222_HUST-main/testcase/input
python3 script.py
```

Script sẽ tạo 5 phiên bản cho mỗi kích thước (N5, N10, N100, N200, N300, N500, N600, N700, N900, N1000).

### 3. Chạy tất cả thuật toán

```bash
cd /Users/apple/TULKH
python3 main.py
```

Chương trình sẽ tự động:
- Tìm tất cả test cases trong folder input (60 test cases)
- Chạy **13 thuật toán** cho mỗi test case:
  - **10 Metaheuristics**: 6 Local Search + 3 Tabu Search + 1 GA
  - **3 Exact algorithms**: Backtracking + Branch&Bound + OR-Tools
- Lưu kết quả vào `result.txt`
- Hiển thị tiến độ và thời gian ước tính

**Lưu ý**: 
- Exact algorithms chạy cho **tất cả** test cases
- Với N > 15, exact algorithms có thể TLE (Time Limit Exceeded)
- Metaheuristics luôn cho kết quả trong thời gian giới hạn

### 4. Xem kết quả

```bash
cat result.txt
```

Format kết quả:
```
============================================================
Test Case: N5
============================================================
LocalSearch_Mode1: 310
LocalSearch_Mode2: 310
LocalSearch_Mode3: 310
LocalSearch_Mode4: 310
LocalSearch_Mode5: 310
LocalSearch_Mode6: 310
TabuSearch_Mode1: 310
TabuSearch_Mode2: 310
TabuSearch_Mode3: 310
GeneticAlgorithm: 310
Backtracking: 310
BranchAndBound: 310
ORTools: 310
```

**Chú thích**:
- Tất cả test cases đều có 13 kết quả
- Exact algorithms có thể trả về "TLE" nếu vượt quá thời gian giới hạn
- Ví dụ: `Backtracking: TLE` hoặc `BranchAndBound: 310`

## Format file test case

Mỗi file test case có format:

```
N
e1 l1 d1
e2 l2 d2
...
eN lN dN
t00 t01 ... t0N
t10 t11 ... t1N
...
tN0 tN1 ... tNN
```

Trong đó:
- `N`: Số lượng khách hàng
- `ei li di`: Earliest time, Latest time, Duration của khách hàng i
- `tij`: Travel time từ vị trí i đến j (i,j = 0..N, với 0 là depot)

## Đặc điểm kỹ thuật

### Đảm bảo tính khả thi
- Tất cả solutions đều được kiểm tra feasibility
- Chỉ chấp nhận solutions thỏa mãn time window constraints
- Nếu không tìm được solution khả thi, thử các strategies khác

### Quản lý thời gian
- Mỗi phương pháp có time limit riêng
- Khi hết thời gian, lấy best solution hiện tại
- Hiển thị progress và estimated time remaining

### Strategies khởi tạo
1. **Nearest**: Chọn customer gần nhất có thể reach được
2. **Earliest**: Sắp xếp theo earliest time window
3. **Latest**: Sắp xếp theo latest time window

## Thống kê

| Kích thước | Số test cases | Thời gian ước tính/test case |
|------------|---------------|------------------------------|
| N5         | 6             | ~5 giây                      |
| N10        | 6             | ~10 giây                     |
| N100       | 6             | ~10 phút                     |
| N200       | 6             | ~20 phút                     |
| N300       | 6             | ~20 phút                     |
| N500       | 6             | ~20 phút                     |
| N600       | 6             | ~30 phút                     |
| N700       | 6             | ~30 phút                     |
| N900       | 6             | ~30 phút                     |
| N1000      | 6             | ~30 phút                     |

**Tổng thời gian ước tính**: ~3-4 giờ cho tất cả 60 test cases

## Yêu cầu hệ thống

- Python 3.6+
- Không cần thư viện bên ngoài (chỉ dùng standard library)
- RAM: Tối thiểu 2GB (khuyến nghị 4GB cho test cases lớn)

## Ghi chú

- Kết quả có thể khác nhau giữa các lần chạy do tính ngẫu nhiên của các thuật toán
- Để có kết quả reproducible, đã set `random.seed(42)` trong main.py
- File `result.txt` sẽ bị ghi đè mỗi khi chạy main.py
- Backup kết quả cũ nếu cần thiết trước khi chạy lại

## Tác giả

Dự án VRPTW Optimization - HUST 2022
