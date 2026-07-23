# 第六章 枚举与模式匹配

## 定义枚举

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

## Option<T> — 替代 null

```rust
enum Option<T> {
    None,
    Some(T),
}

let some_number = Some(5);
let some_char = Some('e');
let absent_number: Option<i32> = None;
```

## match 控制流

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

## if let 简洁控制流

```rust
let config_max = Some(3u8);
if let Some(max) = config_max {
    println!("最大值为 {max}");
}
// 等价于 match config_max { Some(max) => ..., _ => () }
```
