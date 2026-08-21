from waymo_open_dataset.protos import end_to_end_driving_data_pb2


def test_e2e_proto_round_trip():
    frame = end_to_end_driving_data_pb2.E2EDFrame()
    frame.frame.context.name = "run-1"
    frame.frame.timestamp_micros = 1_000_000
    frame.past_states.pos_x.extend([0.0] * 16)

    parsed = end_to_end_driving_data_pb2.E2EDFrame.FromString(
        frame.SerializeToString()
    )

    assert parsed.frame.context.name == "run-1"
    assert parsed.frame.timestamp_micros == 1_000_000
    assert len(parsed.past_states.pos_x) == 16
