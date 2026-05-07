def test_logger_initialization():
    from utils.logger import get_logger
    logger = get_logger("test")
    assert logger is not None


def test_timer_context_manager():
    import time
    from utils.helpers import Timer
    with Timer("test") as t:
        time.sleep(0.01)
    assert t.elapsed >= 0.01


def test_generate_id():
    from utils.helpers import generate_id
    id1 = generate_id("test")
    id2 = generate_id("test")
    assert id1 != id2
    assert id1.startswith("test_")
