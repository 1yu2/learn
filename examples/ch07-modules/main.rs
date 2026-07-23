mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {
            println!("已将顾客添加到候位名单");
        }

        pub fn seat_at_table() {
            println!("请顾客入座");
        }
    }

    mod serving {
        #[allow(dead_code)]
        fn take_order() {}

        #[allow(dead_code)]
        fn serve_order() {}
    }
}

mod back_of_house {
    pub struct Breakfast {
        pub toast: String,
        seasonal_fruit: String,
    }

    impl Breakfast {
        pub fn summer(toast: &str) -> Breakfast {
            Breakfast {
                toast: String::from(toast),
                seasonal_fruit: String::from("桃子"),
            }
        }
    }

    #[allow(dead_code)]
    pub enum Appetizer {
        Soup,
        Salad,
    }
}

use crate::front_of_house::hosting;
use crate::front_of_house::hosting as dinner_hosting;

fn main() {
    // ========== 模块路径 ==========
    println!("=== 模块路径 ===");
    crate::front_of_house::hosting::add_to_waitlist();
    front_of_house::hosting::seat_at_table();

    // ========== use 关键字 ==========
    println!("\n=== use 关键字 ===");
    hosting::add_to_waitlist();

    // ========== as 别名 ==========
    println!("\n=== as 别名 ===");
    dinner_hosting::add_to_waitlist();

    // ========== pub struct ==========
    println!("\n=== pub struct ===");
    let mut meal = back_of_house::Breakfast::summer("黑麦面包");
    meal.toast = String::from("小麦面包");
    println!("我要 {} 吐司", meal.toast);
}
