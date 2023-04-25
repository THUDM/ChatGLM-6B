import sentencepiece as spm
from typing import Tuple

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
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.add_dummy_prefix = add_dummy_prefix

    def put(self, ids: list[int]) -> None:
        self._ids += ids
        self._decode(eos=False)

    def end(self) -> None:
        self._decode(eos=True)
        self._bos_ws_seen = False
        self._nothing_decoded = True
        self._ids = []

    def _decode(self, eos=False) -> None:
        pieces = [self._sp.IdToPiece(i) for i in self._ids]
        consumed = 0
        byte_pieces = []
        for i, piece in enumerate(pieces):
            if not self._sp.IsByte(self._ids[i]):
                self._decoded += ProcessBytePieces(byte_pieces)
                consumed += len(byte_pieces)
                if consumed > 0:
                    self._nothing_decoded = False
                byte_pieces = []
                # if we have seen a bos_ws token or any non-empty token
                if self._bos_ws_seen or (not self._nothing_decoded):
                    self._is_bos_ws = False
                decoded, self._bos_ws_seen = DecodeSentencePiece(
                    piece, self._ids[i], self._is_bos_ws, self._sp)
                self._decoded += decoded
                consumed += 1
                if consumed > 0:
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
