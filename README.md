# Rust 学习文档

> 基于 Rust 官方文档 (The Rust Programming Language) 整理的学习指南。
> 参考版本: Rust 1.90.0, Edition 2024
> 官方文档: https://doc.rust-lang.org/book/

---

## 目录

- [第一章: 快速入门](#第一章-快速入门)
- [第二章: 猜数字游戏](#第二章-猜数字游戏)
- [第三章: 通用编程概念](#第三章-通用编程概念)
- [第四章: 理解所有权](#第四章-理解所有权)
- [第五章: 结构体](#第五章-结构体)
- [第六章: 枚举与模式匹配](#第六章-枚举与模式匹配)
- [第七章: 包、Crate 与模块](#第七章-包crate-与模块)
- [第八章: 常见集合](#第八章-常见集合)
- [第九章: 错误处理](#第九章-错误处理)
- [第十章: 泛型、Trait 与生命周期](#第十章-泛型trait-与生命周期)
- [附录: 学习资源](#附录-学习资源)

---

## 第一章 快速入门

### 安装 Rust

通过 `rustup` 安装 Rust 工具链:

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh

# 验证安装
rustc --version
```

macOS 还需要安装 C 编译器:

```bash
xcode-select --install
```

### Hello, World!

```rust
fn main() {
    println!("Hello, world!");
}
```

编译运行:

```bash
rustc main.rs
./main
```

### Cargo — Rust 的构建系统与包管理器

```bash
# 创建新项目
cargo new hello_cargo
cd hello_cargo

# 构建项目
cargo build

# 构建并运行
cargo run

# 快速检查代码是否能编译 (不生成可执行文件)
cargo check

# 发布构建 (优化)
cargo build --release
```

Cargo.toml 示例:

```toml
[package]
name = "hello_cargo"
version = "0.1.0"
edition = "2024"

[dependencies]
rand = "0.8.5"
```

### 更新 Rust

```bash
rustup update          # 更新 Rust
rustup self uninstall  # 卸载 Rust
rustup doc             # 打开本地文档
```

---

## 第二章 猜数字游戏

一个综合练习项目，涵盖 `let`、`match`、方法、关联函数、外部 crate 等核心概念。

```rust
use rand::Rng;
use std::cmp::Ordering;
use std::io;

fn main() {
    println!("猜数字游戏!");

    let secret_number = rand::thread_rng().gen_range(1..=100);

    loop {
        println!("请输入你的猜测:");

        let mut guess = String::new();

        io::stdin()
            .read_line(&mut guess)
            .expect("读取输入失败");

        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        println!("你猜的数字是: {guess}");

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("太小了!"),
            Ordering::Greater => println!("太大了!"),
            Ordering::Equal => {
                println!("你赢了!");
                break;
            }
        }
    }
}
```

---

## 第三章 通用编程概念

### 变量与可变性

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

### 数据类型

#### 标量类型

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

#### 复合类型

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

### 函数

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

### 控制流

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

---

## 第四章 理解所有权

**所有权是 Rust 最独特的特性**，它使 Rust 无需垃圾回收器即可保证内存安全。

### 所有权规则

1. Rust 中的每个值都有一个 **所有者** (owner)
2. 同一时间只能有 **一个所有者**
3. 当所有者离开作用域，值将被 **丢弃** (drop)

### 栈与堆

- **栈**: LIFO，存储已知固定大小的数据，访问快
- **堆**: 动态分配，通过指针访问，访问相对慢
- 所有权主要管理堆上的数据

### Move 语义

```rust
let s1 = String::from("hello");
let s2 = s1;          // s1 被移动到 s2，s1 不再有效
// println!("{s1}");  // 编译错误! s1 已失效
println!("{s2}");     // 正确
```

### Clone (深拷贝)

```rust
let s1 = String::from("hello");
let s2 = s1.clone();  // 深拷贝堆数据
println!("s1 = {s1}, s2 = {s2}"); // 两者都有效
```

### Copy 类型

实现了 `Copy` trait 的类型，赋值时自动拷贝而非移动:

```rust
let x = 5;
let y = x;
println!("x = {x}, y = {y}"); // 两者都有效
```

`Copy` 类型: 所有整数、`bool`、`f32`/`f64`、`char`、由 `Copy` 类型组成的元组。

### 所有权与函数

```rust
fn main() {
    let s = String::from("hello");
    takes_ownership(s);  // s 移动到函数内
    // println!("{s}");  // 错误!

    let x = 5;
    makes_copy(x);       // i32 是 Copy 类型，x 仍然可用
    println!("{x}");     // 正确
}

fn takes_ownership(some_string: String) {
    println!("{some_string}");
} // some_string 离开作用域，drop 释放内存

fn makes_copy(some_integer: i32) {
    println!("{some_integer}");
}
```

### 引用与借用

```rust
fn main() {
    let s1 = String::from("hello");
    let len = calculate_length(&s1); // 引用传递，不转移所有权
    println!("'{s1}' 的长度是 {len}.");
}

fn calculate_length(s: &String) -> usize {
    s.len()
} // s 是引用，离开作用域不释放内存
```

#### 引用的规则

1. 在任意给定时间，只能拥有以下之一:
   - **一个可变引用** (`&mut T`)
   - **任意数量的不可变引用** (`&T`)
2. 引用必须始终有效 (不允许悬垂引用)

```rust
let mut s = String::from("hello");

let r1 = &s;     // 不可变引用
let r2 = &s;     // 不可变引用，OK
// let r3 = &mut s; // 错误! 不能同时有可变和不可变引用
println!("{r1} {r2}");

let r3 = &mut s;  // r1, r2 不再使用，可以创建可变引用
r3.push_str(" world");
```

### 切片

```rust
let s = String::from("hello world");

let hello = &s[0..5];   // "hello"
let world = &s[6..11];  // "world"
let all = &s[..];       // "hello world"

// 字符串字面量就是切片
let literal: &str = "hello"; // &str 是不可变引用

// 更好的函数签名
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    &s[..]
}
```

---

## 第五章 结构体

### 定义与实例化

```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

let user1 = User {
    active: true,
    username: String::from("alice"),
    email: String::from("alice@example.com"),
    sign_in_count: 1,
};

// 使用 .. 语法从其他实例创建
let user2 = User {
    email: String::from("bob@example.com"),
    ..user1 // user1.username 被移动到 user2，user1 不再可用
};
```

### 元组结构体

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

let black = Color(0, 0, 0);
let origin = Point(0, 0, 0);
```

### 类单元结构体

```rust
struct AlwaysEqual;
let subject = AlwaysEqual;
```

### 方法

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    // 关联函数 (构造函数)
    fn square(size: u32) -> Self {
        Self {
            width: size,
            height: size,
        }
    }

    // 方法 (第一个参数是 &self)
    fn area(&self) -> u32 {
        self.width * self.height
    }

    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}

let rect = Rectangle { width: 30, height: 50 };
println!("面积: {}", rect.area());

let sq = Rectangle::square(10);
```

---

## 第六章 枚举与模式匹配

### 定义枚举

```rust
enum IpAddrKind {
    V4,
    V6,
}

enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

let home = IpAddr::V4(127, 0, 0, 1);
let loopback = IpAddr::V6(String::from("::1"));
```

### Option<T> — 替代 null

```rust
enum Option<T> {
    None,
    Some(T),
}

let some_number = Some(5);
let some_char = Some('e');
let absent_number: Option<i32> = None;
```

### match 控制流

```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}

// 处理 Option<T>
fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
    }
}

// 通配模式
let dice_roll = 9;
match dice_roll {
    3 => add_hat(),
    7 => remove_hat(),
    other => move_player(other), // 绑定值
    // _ => reroll(),            // 不绑定值
    // _ => (),                   // 什么都不做
}
```

### if let 简洁控制流

```rust
let config_max = Some(3u8);
if let Some(max) = config_max {
    println!("最大值为 {max}");
}
// 等价于 match config_max { Some(max) => ..., _ => () }
```

---

## 第七章 包、Crate 与模块

### 模块系统

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
        fn seat_at_table() {}
    }

    mod serving {
        fn take_order() {}
        fn serve_order() {}
        fn take_payment() {}
    }
}

// 使用路径
pub fn eat_at_restaurant() {
    // 绝对路径
    crate::front_of_house::hosting::add_to_waitlist();

    // 相对路径
    front_of_house::hosting::add_to_waitlist();
}
```

### use 关键字

```rust
use crate::front_of_house::hosting;
// use std::collections::HashMap;
// use std::io::{self, Write};
// use std::collections::*; // glob

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

---

## 第八章 常见集合

### Vector

```rust
let mut v: Vec<i32> = Vec::new();
v.push(5);
v.push(6);

let v2 = vec![1, 2, 3];

// 访问元素
let third: &i32 = &v2[2];
let third: Option<&i32> = v2.get(2);

// 遍历
for i in &v2 {
    println!("{i}");
}

for i in &mut v {
    *i += 50; // 解引用
}
```

### 字符串

```rust
let mut s = String::new();
let s1 = "initial contents".to_string();
let s2 = String::from("hello");

s.push_str("bar");
s.push('!');

let s3 = s1 + &s2; // s1 被移动，不再可用

let s4 = format!("{s2}-{s3}"); // 不获取所有权

// 遍历字符
for c in "Зд".chars() {
    println!("{c}");
}
```

### HashMap

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);

let team_name = String::from("Blue");
let score = scores.get(&team_name).copied().unwrap_or(0);

// 遍历
for (key, value) in &scores {
    println!("{key}: {value}");
}

// 只在键不存在时插入
scores.entry(String::from("Blue")).or_insert(50);
```

---

## 第九章 错误处理

Rust 没有异常，将错误分为两类: **可恢复错误** 和 **不可恢复错误**。

### panic! — 不可恢复错误

```rust
fn main() {
    panic!("crash and burn");

    let v = vec![1, 2, 3];
    v[99]; // 索引越界也会 panic!
}
```

设置 `RUST_BACKTRACE=1` 获取回溯信息。

### Result<T, E> — 可恢复错误

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    let greeting_file_result = File::open("hello.txt");

    let greeting_file = match greeting_file_result {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("创建文件失败: {e:?}"),
            },
            other_error => {
                panic!("打开文件失败: {other_error:?}");
            }
        },
    };
}
```

### 快捷方式

```rust
// unwrap: 成功返回值，失败则 panic
let f = File::open("hello.txt").unwrap();

// expect: 类似 unwrap 但可以自定义 panic 信息
let f = File::open("hello.txt").expect("无法打开 hello.txt");
```

### 传播错误

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut username_file = File::open("hello.txt")?;
    let mut username = String::new();
    username_file.read_to_string(&mut username)?;
    Ok(username)
}

// 更简洁的写法
fn read_username_from_file() -> Result<String, io::Error> {
    let mut username = String::new();
    File::open("hello.txt")?.read_to_string(&mut username)?;
    Ok(username)
}

// 最简洁
fn read_username_from_file() -> Result<String, io::Error> {
    fs::read_to_string("hello.txt")
}
```

**`?` 运算符** 相当于: 如果是 `Ok` 则返回内部值，如果是 `Err` 则提前返回错误。

---

## 第十章 泛型、Trait 与生命周期

### 泛型

```rust
// 泛型函数
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// 泛型结构体
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

// 泛型枚举
enum Option<T> { Some(T), None }
enum Result<T, E> { Ok(T), Err(E) }
```

### Trait — 定义共享行为

```rust
pub trait Summary {
    fn summarize(&self) -> String;

    // 默认实现
    fn summarize_author(&self) -> String {
        String::from("(作者未知)")
    }
}

pub struct NewsArticle {
    pub headline: String,
    pub author: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{} - {}", self.headline, self.author)
    }
}

// trait 作为参数
pub fn notify(item: &impl Summary) {
    println!("快讯! {}", item.summarize());
}

// trait bound 语法
pub fn notify<T: Summary>(item: &T) {
    println!("快讯! {}", item.summarize());
}

// 多个 trait bound
pub fn notify(item: &(impl Summary + Display)) {}
pub fn notify<T: Summary + Display>(item: &T) {}

// where 从句
fn some_function<T, U>(t: &T, u: &U) -> i32
where
    T: Display + Clone,
    U: Clone + Debug,
{}

// 返回实现了 trait 的类型
fn returns_summarizable() -> impl Summary {}
```

#### 常用派生 Trait

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MyStruct { /* ... */ }
```

### 生命周期

生命周期确保引用始终有效。

```rust
// 生命周期标注语法
&'a i32      // 带有显式生命周期的引用
&'a mut i32  // 带有显式生命周期的可变引用

// 函数中的生命周期
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// 结构体中的生命周期
struct Excerpt<'a> {
    part: &'a str,
}

// 生命周期省略规则 (三条规则)
// 1. 每个引用参数获得自己的生命周期
// 2. 如果只有一个输入生命周期，则赋给所有输出生命周期
// 3. 如果有 &self 或 &mut self，则其生命周期赋给所有输出生命周期
```

---

## 附录: 学习资源

### 官方资源

| 资源 | 链接 | 说明 |
|------|------|------|
| The Rust Book | https://doc.rust-lang.org/book/ | Rust 圣经，入门必读 |
| Rust by Example | https://doc.rust-lang.org/rust-by-example/ | 通过实例学习 Rust |
| Rustlings | https://github.com/rust-lang/rustlings/ | 交互式命令行练习 |
| 标准库文档 | https://doc.rust-lang.org/std/ | 标准库 API 参考 |
| Cargo 手册 | https://doc.rust-lang.org/cargo/ | 包管理器文档 |
| Rust Reference | https://doc.rust-lang.org/reference/ | 语言参考 |
| Rustonomicon | https://doc.rust-lang.org/nomicon/ | Unsafe Rust 指南 |

### 社区

- 用户论坛: https://users.rust-lang.org
- Discord: https://discord.gg/rust-lang
- 中文社区: https://rustcc.cn

### 其他推荐

- 布朗大学互动版 Rust Book: https://rust-book.cs.brown.edu (带测验、可视化)
- Rust design patterns: https://rust-unofficial.github.io/patterns/

---

## 关键概念速查

| 概念 | 说明 |
|------|------|
| 所有权 (Ownership) | 每个值有唯一的所有者，离开作用域时释放 |
| 借用 (Borrowing) | 通过引用临时使用值而不获取所有权 |
| 生命周期 (Lifetime) | 编译器验证引用有效性 |
| Move 语义 | 赋值/传参会转移所有权 (堆数据) |
| Copy 语义 | 栈数据赋值时自动复制 |
| Trait | 类似接口，定义共享行为 |
| match | 穷尽模式匹配 |
| Result / Option | 无 null/异常，显式处理 |
| unsafe | 绕过部分安全检查 (谨慎使用) |
| Cargo | 官方构建系统与包管理器 |
