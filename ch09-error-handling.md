# 第九章 错误处理

Rust 没有异常，将错误分为两类: **可恢复错误** 和 **不可恢复错误**。

## panic! — 不可恢复错误

```rust
fn main() {
    panic!("crash and burn");

    let v = vec![1, 2, 3];
    v[99]; // 索引越界也会 panic!
}
```

设置 `RUST_BACKTRACE=1` 获取回溯信息。

## Result<T, E> — 可恢复错误

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

## 快捷方式

```rust
// unwrap: 成功返回值，失败则 panic
let f = File::open("hello.txt").unwrap();

// expect: 类似 unwrap 但可以自定义 panic 信息
let f = File::open("hello.txt").expect("无法打开 hello.txt");
```

## 传播错误

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
