# 第五章 结构体

## 定义与实例化

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

## 元组结构体

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

let black = Color(0, 0, 0);
let origin = Point(0, 0, 0);
```

## 类单元结构体

```rust
struct AlwaysEqual;
let subject = AlwaysEqual;
```

## 方法

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
