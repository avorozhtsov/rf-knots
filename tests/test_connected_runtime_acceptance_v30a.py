import pytest
from check_connected_runtime_v30a import verify_started

def test_saved_metadata_not_progress():
    with pytest.raises(AssertionError):verify_started(2017313,2017313,'VERIFIED_ACTIVE',2017313)

def test_not_yet_verified():
    with pytest.raises(AssertionError):verify_started(2017400,2017313,'LAUNCHED_AWAITING_PROGRESS',2017313)

def test_real_continuation():
    verify_started(2017400,2017313,'VERIFIED_ACTIVE',2017313)
