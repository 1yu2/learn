#[derive(Debug)]
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

impl Message {
    fn call(&self) {
        match self {
            Message::Quit => println!("退出"),
            Message::Move { x, y } => println!("移动到 ({x}, {y})"),
            Message::Write(text) => println!("写入: {text}"),
            Message::ChangeColor(r, g, b) => println!("改变颜色: ({r}, {g}, {b})"),
        }
    }
}

#[derive(Debug)]
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: &Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}

fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
    }
}

fn main() {
    // ========== 枚举 ==========
    println!("=== 枚举 ===");
    let home = IpAddr::V4(127, 0, 0, 1);
    let loopback = IpAddr::V6(String::from("::1"));
    println!("home: {home:?}");
    println!("loopback: {loopback:?}");

    let msgs = [
        Message::Quit,
        Message::Move { x: 10, y: 20 },
        Message::Write(String::from("hello")),
        Message::ChangeColor(255, 0, 0),
    ];
    for msg in &msgs {
        msg.call();
    }

    // ========== Option<T> ==========
    println!("\n=== Option<T> ===");
    let some_number = Some(5);
    let absent_number: Option<i32> = None;
    println!("some_number: {some_number:?}, absent: {absent_number:?}");

    // ========== match ==========
    println!("\n=== match ===");
    let coin = Coin::Quarter;
    println!("{coin:?} 的价值: {} 美分", value_in_cents(&coin));

    let five = Some(5);
    let six = plus_one(five);
    let none = plus_one(None);
    println!("five: {five:?}, six: {six:?}, none: {none:?}");

    // ========== if let ==========
    println!("\n=== if let ===");
    let config_max = Some(3u8);
    if let Some(max) = config_max {
        println!("最大值为 {max}");
    }

    let config_none: Option<u8> = None;
    if let Some(max) = config_none {
        println!("最大值为 {max}");
    } else {
        println!("没有配置值");
    }
}
