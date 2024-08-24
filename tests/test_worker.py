import pytest

from eduadvisor.worker import clean_and_parse


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        (
            '[snippet: "example", title: "title1", link: "link1"], [snippet: "example2", title: "title2", link: "link2"]',
            [
                {"snippet": '"example"', "title": '"title1"', "link": '"link1"'},
                {"snippet": '"example2"', "title": '"title2"', "link": '"link2"'},
            ],
        ),
        (
            "[snippet: Josephine and Napoleon cared for each other until death . According to PBS, Napoleon, now about 40 years old, immediately began searching for a new wife. His ideal candidate was Anna Pavlovna, the ..., title: Napoleon and Josephine Had a Stormy, Unfaithful 13-Year Marriage, link: https://www.biography.com/royalty/a45836725/napoleon-josephine-relationship-marriage-divorce], [snippet: Josephine and Napoleon cared for each other until death . According to PBS, Napoleon, now about 40 years old, immediately began searching for a new wife. His ideal candidate was Anna Pavlovna, the ..., title: Napoleon and Josephine Had a Stormy, Unfaithful 13-Year Marriage, link: https://www.biography.com/royalty/a45836725/napoleon-josephine-relationship-marriage-divorce]",
            [
                {
                    "snippet": "Josephine and Napoleon cared for each other until death . According to PBS, Napoleon, now about 40 years old, immediately began searching for a new wife. His ideal candidate was Anna Pavlovna, the ...",
                    "title": "Napoleon and Josephine Had a Stormy, Unfaithful 13-Year Marriage",
                    "link": "https://www.biography.com/royalty/a45836725/napoleon-josephine-relationship-marriage-divorce",
                },
                {
                    "snippet": "Josephine and Napoleon cared for each other until death . According to PBS, Napoleon, now about 40 years old, immediately began searching for a new wife. His ideal candidate was Anna Pavlovna, the ...",
                    "title": "Napoleon and Josephine Had a Stormy, Unfaithful 13-Year Marriage",
                    "link": "https://www.biography.com/royalty/a45836725/napoleon-josephine-relationship-marriage-divorce",
                },
            ],
        ),
    ],
)
def test_clean_and_parse(input_string: str, expected_output: list[dict[str, str]]):
    result = clean_and_parse(input_string)
    assert result == expected_output
