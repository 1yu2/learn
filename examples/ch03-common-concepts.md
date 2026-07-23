# 第三章 通用编程概念

## 变量与可变性

```rust
// 默认不可变
let x = 5;
// x = 6; // 编译错误!

// 可变变量
let mut y = 5;
y = 6; // 正确

// 常量
const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;

// 遮蔽 (Shadowing)
let z = 5;
let z = z + 1; // z = 6，新的变量，可以改变类型
```

## 数据类型

### 标量类型

| 类型 | 描述 |
|------|------|
| `i8`, `i16`, `i32`, `i64`, `i128`, `isize` | 有符号整数 |
| `u8`, `u16`, `u32`, `u64`, `u128`, `usize` | 无符号整数 |
| `f32`, `f64` | 浮点数 (默认 `f64`) |
| `bool` | 布尔值 `true` / `false` |
| `char` | Unicode 字符 (4 字节) |

```rust
let a: u32 = 42;
let b = 3.14;        // f64
let c: bool = true;
let d = '🦀';        // char, 单引号
```

### 复合类型

```rust
// 元组 (Tuple)
let tup: (i32, f64, u8) = (500, 6.4, 1);
let (x, y, z) = tup;
let five_hundred = tup.0;

// 数组 (Array) — 固定长度，栈上分配
let arr: [i32; 5] = [1, 2, 3, 4, 5];
let zeros = [0; 5];  // [0, 0, 0, 0, 0]
let first = arr[0];
```

## 函数

```rust
fn main() {
    let result = add(5, 3);
    println!("结果: {result}");
}

fn add(x: i32, y: i32) -> i32 {
    x + y // 表达式，不加分号表示返回值
}
```

- **语句**: 执行操作但不返回值 (以分号结尾)
- **表达式**: 计算并返回值 (不以分号结尾)

## 控制流

```rust
// if 表达式
let number = 6;
if number % 4 == 0 {
    println!("number 能被 4 整除");
} else if number % 3 == 0 {
    println!("number 能被 3 整除");
} else {
    println!("number 不能被 4 或 3 整除");
}

// if 是表达式，可用于赋值
let condition = true;
let number = if condition { 5 } else { 6 };

// 循环
loop {
    // 无限循环
    break;
}

let mut counter = 0;
let result = loop {
    counter += 1;
    if counter == 10 {
        break counter * 2; // 从 loop 返回值
    }
};

// while 循环
let mut number = 3;
while number != 0 {
    println!("{number}!");
    number -= 1;
}

// for 循环
let arr = [10, 20, 30, 40, 50];
for element in arr {
    println!("值: {element}");
}

// Range
for number in (1..4).rev() {
    println!("{number}!");
}
```
