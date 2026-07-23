# 第四章 理解所有权

**所有权是 Rust 最独特的特性**，它使 Rust 无需垃圾回收器即可保证内存安全。

## 所有权规则

1. Rust 中的每个值都有一个 **所有者** (owner)
2. 同一时间只能有 **一个所有者**
3. 当所有者离开作用域，值将被 **丢弃** (drop)

## 栈与堆

- **栈**: LIFO，存储已知固定大小的数据，访问快
- **堆**: 动态分配，通过指针访问，访问相对慢
- 所有权主要管理堆上的数据

## Move 语义

```rust
let s1 = String::from("hello");
let s2 = s1;          // s1 被移动到 s2，s1 不再有效
// println!("{s1}");  // 编译错误! s1 已失效
println!("{s2}");     // 正确
```

## Clone (深拷贝)

```rust
let s1 = String::from("hello");
let s2 = s1.clone();  // 深拷贝堆数据
println!("s1 = {s1}, s2 = {s2}"); // 两者都有效
```

## Copy 类型

实现了 `Copy` trait 的类型，赋值时自动拷贝而非移动:

```rust
let x = 5;
let y = x;
println!("x = {x}, y = {y}"); // 两者都有效
```

`Copy` 类型: 所有整数、`bool`、`f32`/`f64`、`char`、由 `Copy` 类型组成的元组。

## 所有权与函数

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

## 引用与借用

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

### 引用的规则

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

## 切片

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
