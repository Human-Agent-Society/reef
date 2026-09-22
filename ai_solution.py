To fix the issue, we need to ensure that each value in the `rewards` map is validated in the same way as the scalar `reward` field. This is achieved by using the `checked_optional_number` function for each value in the rewards.

Here is the revised code:

```python
def checked_optional_number(value, field_name):
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as e:
        raise WireError(f"{field_name}: {value} is not a number") from e

def play_from_document(document):
    if not document:
        return None
    try:
        rewards = document.get("rewards", {})
        rewards = {str(name): checked_optional_number(value, name) for name, value in rewards.items()}
        return play_document({
            "play": document["play"],
            "rewards": rewards,
            "reward": checked_optional_number(document.get("reward"), "reward")
        })
    except WireError as e:
        raise e
    except Exception as e:
        raise WireError("Error in play_from_document") from e
```

The rewards are now validated using `checked_optional_number` for each key-value pair.