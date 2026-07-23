# 第七章 包、Crate 与模块

## 模块系统

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

## use 关键字

```rust
use crate::front_of_house::hosting;
// use std::collections::HashMap;
// use std::io::{self, Write};
// use std::collections::*; // glob

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```
