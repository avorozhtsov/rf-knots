from build_normalized_random_roster_v3 import move_choices
from rf_knots import reference
from rf_knots.actions import ActionSpec,COMMUTE,BRAID
from rf_knots.embedding_normalization import fast_normalize_global_view as norm


def test_legal_choices_are_involutions_and_normalized_new():
    spec=ActionSpec(20,5);word=(1,3,1,2,1,3,2)
    choices=move_choices(word,4,spec,{norm(word,4)})
    assert choices
    for action,after in choices:
        assert spec.decode(action)[0] in [COMMUTE,BRAID]
        assert reference.is_legal(spec,*after,action,False)
        assert reference.apply(spec,*after,action)==(word,4)
        assert norm(*after)!=norm(word,4)


def test_padding_erased_not_counted_as_diversity():
    assert norm((1,-1,1,1,1),2)==norm((1,1,1),2)
