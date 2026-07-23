package main

import "fmt"

func main() {
	fmt.Println("Hello, Go!")
}

// 变量声明
func varExample() {
	var name string = "Go"
	var age = 10
	language := "Go" // 短变量声明（仅函数内可用）

	fmt.Println(name, age, language)
}

// 基本类型
func typeExample() {
	var (
		flag    bool       = true
		integer int        = 42
		float   float64    = 3.14
		text    string     = "hello"
		bt      byte       = 'A'
		ru      rune       = '世'
	)

	fmt.Println(flag, integer, float, text, bt, ru)

	// 零值
	var i int     // 0
	var s string  // ""
	var b bool    // false
	fmt.Printf("zero values: %v, %q, %v\n", i, s, b)
}
