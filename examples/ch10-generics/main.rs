use std::fmt::Display;

// ========== 泛型 ==========

fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

struct Point<T> {
    x: T,
    y: T,
}

impl<T: Display> Point<T> {
    fn display(&self) {
        println!("Point({}, {})", self.x, self.y);
    }
}

// ========== Trait ==========

trait Summary {
    fn summarize(&self) -> String;

    fn summarize_author(&self) -> String {
        String::from("(未知作者)")
    }
}

struct NewsArticle {
    headline: String,
    author: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{} - {}", self.headline, self.author)
    }
}

struct Tweet {
    username: String,
    content: String,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("@{}: {}", self.username, &self.content[..50.min(self.content.len())])
    }
}

fn notify(item: &impl Summary) {
    println!("快讯! {}", item.summarize());
}

fn notify_bound<T: Summary + Display>(item: &T) {
    println!("通知: {}", item.summarize());
}

impl Display for NewsArticle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.headline)
    }
}

// ========== 生命周期 ==========

fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

struct Excerpt<'a> {
    part: &'a str,
}

impl<'a> Excerpt<'a> {
    fn display(&self) {
        println!("摘录: {}", self.part);
    }
}

fn main() {
    // ========== 泛型 ==========
    println!("=== 泛型 ===");
    let number_list = vec![34, 50, 25, 100, 65];
    println!("最大数: {}", largest(&number_list));

    let char_list = vec!['y', 'm', 'a', 'q'];
    println!("最大字符: {}", largest(&char_list));

    let p = Point { x: 1, y: 2 };
    p.display();

    // ========== Trait ==========
    println!("\n=== Trait ===");
    let article = NewsArticle {
        headline: String::from("Rust 1.90 发布!"),
        author: String::from("Rust 团队"),
    };
    println!("{}", article.summarize());

    let tweet = Tweet {
        username: String::from("rustlang"),
        content: String::from("Rust 是一门注重安全、并发和性能的系统编程语言。"),
    };
    notify(&tweet);
    notify_bound(&article);

    // ========== 生命周期 ==========
    println!("\n=== 生命周期 ===");
    let s1 = String::from("短");
    let s2 = String::from("较长的字符串");
    println!("较长的是: {}", longest(&s1, &s2));

    let text = String::from("这是一段重要的文本。剩下的内容可以忽略。");
    let excerpt = Excerpt {
        part: &text[..12],
    };
    excerpt.display();
}
