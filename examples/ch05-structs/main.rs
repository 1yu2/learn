#[derive(Debug)]
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn square(size: u32) -> Self {
        Self {
            width: size,
            height: size,
        }
    }

    fn area(&self) -> u32 {
        self.width * self.height
    }

    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}

struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    // ========== 结构体定义与实例化 ==========
    println!("=== 结构体 ===");
    let user1 = User {
        active: true,
        username: String::from("alice"),
        email: String::from("alice@example.com"),
        sign_in_count: 1,
    };
    println!("user1: {user1:?}");

    let user2 = User {
        email: String::from("bob@example.com"),
        ..user1
    };
    println!("user2: {user2:?}");

    // ========== 元组结构体 ==========
    println!("\n=== 元组结构体 ===");
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
    println!("black = ({}, {}, {})", black.0, black.1, black.2);
    println!("origin = ({}, {}, {})", origin.0, origin.1, origin.2);

    // ========== 方法 ==========
    println!("\n=== 方法 ===");
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };
    println!("矩形 {:?} 的面积是 {}", rect1, rect1.area());

    let rect2 = Rectangle {
        width: 20,
        height: 40,
    };
    println!("rect1 能容纳 rect2 吗? {}", rect1.can_hold(&rect2));

    let sq = Rectangle::square(10);
    println!("正方形 {:?} 的面积是 {}", sq, sq.area());
}
