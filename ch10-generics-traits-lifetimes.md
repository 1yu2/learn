# 第十章 泛型、Trait 与生命周期

## 泛型

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

## Trait — 定义共享行为

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

### 常用派生 Trait

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MyStruct { /* ... */ }
```

## 生命周期

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
