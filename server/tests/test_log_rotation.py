from trainer.log_rotation import RotatingTextStream


def test_rotating_stream_bounds_log_history(tmp_path) -> None:
    path = tmp_path / "dashboard-runtime.err"
    stream = RotatingTextStream(path, max_bytes=64, backups=2)
    try:
        for index in range(20):
            stream.write(f"error-{index:02d}-payload\n")
    finally:
        stream.close()

    logs = [path, path.with_name(f"{path.name}.1"), path.with_name(f"{path.name}.2")]
    assert all(item.stat().st_size <= 64 for item in logs if item.exists())
    assert sum(item.stat().st_size for item in logs if item.exists()) <= 64 * 3