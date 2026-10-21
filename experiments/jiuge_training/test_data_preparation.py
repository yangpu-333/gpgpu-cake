import unittest

from prepare_alpaca_cpt import document


class DataPreparationTests(unittest.TestCase):
    def test_whole_instruction_input_and_answer_are_preserved(self):
        self.assertEqual(document({'instruction': '问题', 'input': '材料', 'output': '回答'}),
                         '### 指令：\n问题\n### 输入：\n材料\n### 回答：\n回答')

    def test_optional_empty_input_and_invalid_fields(self):
        self.assertEqual(document({'instruction': '问题', 'input': '', 'output': '回答'}),
                         '### 指令：\n问题\n### 回答：\n回答')
        with self.assertRaises(ValueError):
            document({'instruction': '问题', 'input': None, 'output': '回答'})


if __name__ == '__main__':
    unittest.main()
