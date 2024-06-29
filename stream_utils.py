import sentencepiece as spm
from typing import Tuple
import re
import unittest

# python implantation of https://github.com/google/sentencepiece/blob/master/src/sentencepiece_processor.cc


def DecodeSentencePiece(piece: str, id: int, is_bos_ws: bool, sp: spm.SentencePieceProcessor, add_dummy_prefix=True, remove_extra_whitespaces=False) -> Tuple[str, bool]:
    '''
    Returns decoded piece and a boolean indicating if the function has consumed
    a bos whitespace token (a piece starting with a kSpaceSymbol). This is used
    to strip only the first whitespace token from the decoded sequence for
    add_dummy_prefix.
    '''
    if sp.IsControl(id):  # <s>, </s>
        return "", False  # invisible symbol.
    elif sp.IsUnknown(id):
        if sp.IdToPiece(id) == piece:  # <unk>
            return SPStreamDecoder.DefaultUnknownSymbol, False
        else:  # return piece when piece is not <unk>.
            return piece, False
    has_bos_ws = False  # whether the token starts with a kSpaceSymbol
    # Consume if the current position is bos and
    # piece starts with kSpaceSymbol.
    if is_bos_ws and (add_dummy_prefix or remove_extra_whitespaces):
        t = piece.removeprefix(SPStreamDecoder.SpaceSymbol)
        has_bos_ws = t != piece
        piece = t
        # if we are removing extra whitespace, we remove all leading whitespace
        if remove_extra_whitespaces:
            has_bos_ws = False
    return piece.replace(SPStreamDecoder.SpaceSymbol, " "), has_bos_ws


def ProcessBytePieces(pieces: list[str]) -> str:
    '''
    Modified version of original code
    '''
    if len(pieces) == 0:
        return ""
    surfaces = ""
    # Constructs byte sequence.
    bytes_ = bytes([int(piece[1:-1], base=16) for piece in pieces])
    # Set surfaces of `bytes` for each Unicode character.
    while len(bytes_) > 0:
        try:
            surfaces += bytes_.decode('utf-8')
            break
        except UnicodeDecodeError as e:
            # The byte piece at `e.start` is structurally invalid. Map it to
            # REPLACEMENT CHARACTER (U+FFFD).
            surfaces += bytes_[:e.start].decode('utf-8')
            surfaces += SPStreamDecoder.ReplacementCharacter
            bytes_ = bytes_[e.end:]
            continue
    return surfaces


class SPStreamDecoder:
    SpaceSymbol = chr(0x2581)
    DefaultUnknownSymbol = chr(0x2047)
    ReplacementCharacter = chr(0xFFFD)

    def __init__(self, sp: spm.SentencePieceProcessor, remove_extra_whitespaces=False, add_dummy_prefix=True) -> None:
        self._sp = sp
        self._bos_ws_seen = False
        # 'is_bos_ws': whether we expect a bos ws token to consume.
        self._is_bos_ws = True
        self._nothing_decoded = True
        self._ids = []
        self._decoded = ""
        self._ending = False
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.add_dummy_prefix = add_dummy_prefix

    def put(self, ids: list[int]) -> None:
        self._ending = False
        self._ids += ids
        self._decode(eos=False)

    def end(self) -> None:
        self._decode(eos=True)
        self._is_bos_ws = True
        self._bos_ws_seen = False
        self._nothing_decoded = True
        self._ending = True
        self._ids = []

    def _decode(self, eos=False) -> None:
        pieces = [self._sp.IdToPiece(i) for i in self._ids]
        consumed = 0
        byte_pieces = []
        for i, piece in enumerate(pieces):
            if not self._sp.IsByte(self._ids[i]):
                self._decoded += ProcessBytePieces(byte_pieces)
                consumed += len(byte_pieces)
                if len(self._decoded) > 0:
                    self._nothing_decoded = False
                byte_pieces = []
                # if we have seen a bos_ws token or any non-empty token
                if self._bos_ws_seen or (not self._nothing_decoded):
                    self._is_bos_ws = False
                decoded, self._bos_ws_seen = DecodeSentencePiece(
                    piece, self._ids[i], self._is_bos_ws, self._sp)
                self._decoded += decoded
                consumed += 1
                if len(self._decoded) > 0:
                    self._nothing_decoded = False
            else:
                byte_pieces.append(piece)
        if eos:
            self._decoded += ProcessBytePieces(byte_pieces)
        else:
            self._ids = self._ids[consumed:]

    def get(self) -> str:
        t = self._decoded
        self._decoded = ""
        return t


class ChatGLMStreamDecoder(SPStreamDecoder):

    def get(self) -> str:
        # if prefix of special tokens found, wait till it's impossible or end of decode
        if "[" in self._decoded and len(self._decoded)-self._decoded.index("[") < 8 and not self._ending:
            return ""
        if "<" in self._decoded and len(self._decoded)-self._decoded.index("<") < 12 and not self._ending:
            return ""
        self._ending = False
        t = self._decoded
        self._decoded = ""
        t = t.replace("<n>", "\n")
        t = t.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            t = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], t)
            t = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], t)
        # for i in range(max_len, 1, -1):
        #    t = t.replace(f"<|blank_{i}|>", " " * i)
        for blank_token in re.findall(r"<\|blank_\d+\|>", t):
            t = t.replace(blank_token, " " *
                          int(re.search(r"\d+", blank_token)[0]))
        return t


class ChatGLMStreamDecoderTest(unittest.TestCase):
    def test_ChatGLM_StreamDecoder(self):
        from transformers import AutoTokenizer, AutoModel
        test_strings = [
            "你好👋",  # multi-byte encoding
            "Hello this is ChatGLM!",  # normal text
            "你好👋 This is ChatGLM!",  # multi-byte encoding with tail
            "!?.,！？。，",  # punctuations
            "A\nB",  # "<n>" -> "\n"
            "[[训练时间]]",  # training time token
            "[[训练时间]123",  # broken training time token
            "1        1",  # blank token. Note: It's hard to match the results of strip(), so add leading and tailing "1"
            "<|blank_8|123",  # broken blank token
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained(
            "THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
        model = model.eval()
        encoded_ids = [tokenizer(x)['input_ids'] for x in test_strings]
        stream_decoder = ChatGLMStreamDecoder(
            tokenizer.sp_tokenizer.text_tokenizer.sp)
        # original output
        expected_outputs = [model.process_response(
            tokenizer.decode(x)) for x in encoded_ids]
        # decode token by token
        decoded_strings_stream_token_by_token = [None for _ in test_strings]
        for i in range(len(test_strings)):
            res = []
            for t in encoded_ids[i]:
                stream_decoder.put([t])
                res.append(stream_decoder.get())
            stream_decoder.end()
            res.append(stream_decoder.get())
            res = "".join(res)
            decoded_strings_stream_token_by_token[i] = res
        # decode all at once
        decoded_strings_stream = [None for _ in test_strings]
        for i in range(len(test_strings)):
            stream_decoder.put(encoded_ids[i])
            stream_decoder.end()
            decoded_strings_stream[i] = stream_decoder.get()
        for i in range(len(test_strings)):
            print(
                f"Stream decoder test{i}: expected: '{expected_outputs[i]}', token_by_token: '{decoded_strings_stream_token_by_token[i]}', all at once: '{decoded_strings_stream[i]}'")
            self.assertEqual(
                expected_outputs[i], decoded_strings_stream_token_by_token[i])
            self.assertEqual(expected_outputs[i], decoded_strings_stream[i])


if __name__ == "__main__":
    unittest.main()
