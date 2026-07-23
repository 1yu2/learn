# Go 核心概念

## 语言特性

Go 是由 Google 开发的静态类型、编译型开源编程语言。

### 设计哲学

- **简洁**：语法简单，关键字只有 25 个，没有复杂的特性。
- **高效**：编译快，执行快，开发快。
- **并发**：Goroutine 和 Channel 提供简洁的并发模型。
- **组合优于继承**：通过结构体嵌入（Embedding）和接口（Interface）实现代码复用。

### 编译与运行

Go 是编译型语言，源代码编译为机器码二进制文件：

```bash
go run main.go           # 编译并运行
go build main.go         # 编译为可执行文件
go build -o myapp .      # 指定输出文件名
```

Go 编译速度极快，编译后的二进制文件包含运行所需的一切（包括 Go 运行时），无需外部依赖。

### 垃圾回收

Go 自带垃圾回收器（Garbage Collector），自动管理内存，无需手动分配/释放内存。

## 包与模块

### Package（包）

- 每个 Go 文件属于一个包（package），文件第一行声明包名。
- 同一个目录下所有 `.go` 文件必须属于同一个包。
- `main` 包包含程序的入口函数 `main()`。
- 大写字母开头的标识符是导出的（Exported），可以被其他包使用。
- 小写字母开头的标识符是未导出的，仅在当前包内可见。

### Module（模块）

- Module 是 Go 的依赖管理单元，由 `go.mod` 文件定义。
- 模块包含一个或多个包。
- Go 1.11 引入 Modules，Go 1.16 起成为默认模式。

```go
module github.com/user/project   // 模块路径
go 1.22                          // Go 版本
```

### 常用命令

```bash
go mod init <module-path>     # 初始化模块
go mod tidy                   # 清理和更新依赖
go get <package>              # 添加依赖
go mod edit -go <version>     # 修改 Go 版本
go mod vendor                 # 创建 vendor 目录
```

## 基础语法概览

### 变量

```go
var name string = "Go"       // 完整声明
var name = "Go"              // 类型推导
name := "Go"                 // 短变量声明（仅函数内可用）
```

### 类型

```go
bool                         // true / false
int, int8, int16, int32, int64
uint, uint8, uint16, uint32, uint64
float32, float64
string                       // UTF-8 字符串
byte                         // uint8 别名
rune                         // int32 别名，表示 Unicode 码点
complex64, complex128        // 复数
```

### 零值（Zero Values）

没有显式赋值的变量会被赋予零值：
- 数值类型：`0`
- 布尔类型：`false`
- 字符串：`""`
- 指针、切片、映射、通道、函数、接口：`nil`

### 控制流

```go
// if
if x > 0 {
    // ...
} else if x < 0 {
    // ...
} else {
    // ...
}

// for (唯一的循环语句)
for i := 0; i < 10; i++ {}  // 经典 for
for sum < 100 {}             // while 风格
for {}                       // 无限循环
for i, v := range slice {}  // 遍历

// switch
switch x {
case 1:
    // ...
default:
    // ...
}
```

### defer

`defer` 延迟函数执行到外层函数返回之前。多个 defer 按 LIFO 顺序执行。

常用场景：关闭文件、释放锁、捕获 panic。

## 数组、切片与映射

### 数组（Array）

- 固定长度，是值类型。
- `[3]int` 和 `[5]int` 是完全不同的类型。

```go
var arr [5]int
arr := [3]int{1, 2, 3}
arr := [...]int{1, 2, 3}   // 编译器推导长度
```

### 切片（Slice）

- 动态长度的"视图"，底层引用数组。
- 比数组更常用，是 Go 的核心数据结构。

```go
s := make([]int, 5)         // 长度 5，容量 5
s := make([]int, 0, 10)     // 长度 0，容量 10
s := []int{1, 2, 3}         // 切片字面量
s = append(s, 4)            // 追加元素

// 切片操作
s[1:3]                       // 索引 1 到 2（不含 3）
s[:]                         // 整个切片
```

### 映射（Map）

- 无序的键值对集合。
- nil map 不可写入，需 make 初始化。

```go
m := make(map[string]int)
m["key"] = 42
v, ok := m["key"]            // 检查键是否存在
delete(m, "key")
```

## 函数与方法

### 函数

```go
func add(a, b int) int {
    return a + b
}

// 多返回值
func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// 命名返回值
func split(sum int) (x, y int) {
    x = sum * 4 / 9
    y = sum - x
    return    // 裸返回
}
```

### 方法

方法是带接收者（Receiver）的函数：

```go
type Counter struct {
    value int
}

// 值接收者
func (c Counter) Value() int {
    return c.value
}

// 指针接收者（可以修改接收者）
func (c *Counter) Inc() {
    c.value++
}
```

## 结构体与接口

### 结构体

Go 没有类，用结构体组织数据：

```go
type Person struct {
    Name string
    Age  int
}

// 嵌入（组合）
type Employee struct {
    Person    // 匿名嵌入
    Company string
}
```

### 接口

接口定义行为（方法集），类型隐式实现接口：

```go
type Writer interface {
    Write([]byte) (int, error)
}
```

任何实现了 `Write([]byte) (int, error)` 方法的类型都实现了 `Writer` 接口。

### 空接口

`interface{}`（即 `any`）可以表示任意类型的值。

## 并发

### Goroutine

Goroutine 是由 Go 运行时管理的轻量级线程：

```go
go func() {
    // 并发执行的代码
}()
```

Goroutine 非常轻量（几 KB 的栈空间），可以创建成千上万个。

### Channel

Channel 用于 Goroutine 之间的通信：

```go
ch := make(chan int)        // 无缓冲 Channel
ch := make(chan int, 10)    // 有缓冲 Channel

ch <- 42                    // 发送
v := <-ch                   // 接收
close(ch)                   // 关闭

// 遍历 Channel（直到关闭）
for v := range ch {
    // ...
}
```

### Select

Select 用于多路 Channel 操作：

```go
select {
case v := <-ch1:
    // ch1 有数据
case ch2 <- 42:
    // 数据发送到 ch2
case <-time.After(1 * time.Second):
    // 超时
default:
    // 非阻塞
}
```

### Mutex

`sync.Mutex` 提供互斥锁：

```go
var mu sync.Mutex
mu.Lock()
// 临界区
mu.Unlock()
```

### Context

Context 用于跨 API 边界传递取消信号、超时和请求范围的值：

```go
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

select {
case <-ctx.Done():
    // 超时或取消
}
```

## 错误处理

Go 没有 try-catch 异常机制，通过返回值处理错误：

```go
f, err := os.Open("file.txt")
if err != nil {
    log.Fatal(err)
}
defer f.Close()
```

### 错误包装（Go 1.13+）

```go
err := fmt.Errorf("open file: %w", err)    // 包装错误
original := errors.Unwrap(err)              // 解包
errors.Is(err, os.ErrNotExist)              // 检查
var pathError *os.PathError
errors.As(err, &pathError)                  // 类型断言
```

## 并发哲学

> Don't communicate by sharing memory; share memory by communicating.
> — Go Proverbs

Go 鼓励使用 Channel 传递数据的所有权，而不是通过共享内存加锁来通信。

## 工具链

| 命令 | 用途 |
|------|------|
| `go build` | 编译 |
| `go run` | 编译并运行 |
| `go test` | 运行测试 |
| `go fmt` | 格式化代码 |
| `go vet` | 静态分析 |
| `go mod` | 模块管理 |
| `go doc` | 查看文档 |
| `go get` | 安装依赖 |
| `go install` | 编译并安装到 $GOPATH/bin |
| `pprof` | 性能分析 |

## 参考链接

- [Go 官网](https://go.dev/)
- [A Tour of Go](https://go.dev/tour/)
- [Effective Go](https://go.dev/doc/effective_go)
- [Language Specification](https://go.dev/ref/spec)
- [Standard Library](https://pkg.go.dev/std)
