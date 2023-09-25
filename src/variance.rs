#[cfg(test)]
mod variance_test {
    #[allow(dead_code)]
    pub fn lifetime_align() {
        let hello: &'static str = "hello";
        {
            let world = String::from("world");
            let world = &world; // 'world has a shorter lifetime than 'static
            debug(hello, world);
        }
    }

    #[allow(dead_code)]
    // Note: debug expects two parameters with the *same* lifetime
    fn debug<'a>(a: &'a str, b: &'a str) {
        println!("a = {a:?} b = {b:?}");
    }

    #[allow(dead_code)]
    pub fn variance_problem() {
        let mut hello: &'static str = "hello";
        {
            let world = String::from("world");
            // assign(&mut hello, &world);
        }
        println!("{hello}"); // use after free 😿
    }

    // 实际参数: &mut &'static str , &'a str
    // 因为签名是 &mut T, T， 所以需要做一层转换，将 &mut &'static str > &mut &'a str
    // 但是对于这种类型的转换，&mut T 是 invariant, 也就是不允许对 T 进行向下转换
    // 这也是上面会编译失败的原因
    #[allow(dead_code)]
    fn assign<T>(input: &mut T, val: T) {
        *input = val;
    }
}
