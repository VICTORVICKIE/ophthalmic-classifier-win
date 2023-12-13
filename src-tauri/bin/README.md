# Copy these for Tauri SideCar

```bash
cd oct-tf && cargo build --release
cp oct-tf/target/release/oct-tf.exe ./oct-tf-x86_64-pc-windows-msvc.exe
cp oct-tf/target/tensorflow-sys*/out/tensorflow* .
```