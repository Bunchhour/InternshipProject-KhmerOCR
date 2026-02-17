import torch

class KhmerLabelConverter:
    """
    Manages the vocabulary for the OCR model.
    Converts 'ក' -> 5 (Encoding) and 5 -> 'ក' (Decoding).
    """
    def __init__(self, chars):
        # Sort characters to ensure deterministic ID assignment
        self.chars = sorted(list(set(chars)))
        
        # 0 is reserved for the CTC 'blank' token
        # So our characters start from index 1
        self.char_to_id = {c: i + 1 for i, c in enumerate(self.chars)}
        self.id_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        
        # The blank token is always 0
        self.blank_id = 0
    
    def encode(self, text):
        """
        Converts a text string into a list of IDs.
        Example: "កខ" -> tensor([10, 11])
        """
        # Remove Zero Width Space (common in Khmer text but useless for OCR)
        text = text.replace('\u200b', '') 
        
        encoded = []
        for char in text:
            if char in self.char_to_id:
                encoded.append(self.char_to_id[char])
            else:
                # Optionally handle unknown characters here
                # For now, we just skip them
                continue
        
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, current_ids):
        """
        Converts a list of IDs back to a string.
        filters out the CTC blank token (0).
        """
        res = ""
        for i in current_ids:
            # Skip the blank token
            if i == self.blank_id:
                continue
            
            # Convert ID back to character
            # We check if 'i' is a tensor or int to be safe
            idx = i.item() if isinstance(i, torch.Tensor) else i
            res += self.id_to_char.get(idx, "")
            
        return res

    def get_num_classes(self):
        """
        Returns total classes = (Number of unique chars) + 1 (Blank)
        Used to define the final layer size of the model.
        """
        return len(self.chars) + 1
    
# ======= Code Testing ===================
if __name__ == "__main__":
    # Example Khmer characters (you can expand this)
    khmer_chars = "កខគឃងចឆជញ"

    # Initialize converter
    converter = KhmerLabelConverter(khmer_chars)

    # ---- BASIC INFO ----
    print("Characters:", converter.chars)
    print("Char → ID:", converter.char_to_id)
    print("ID → Char:", converter.id_to_char)
    print("Blank ID:", converter.blank_id)
    print("Number of classes:", converter.get_num_classes())
    print("-" * 40)

    # ---- ENCODE TEST ----
    text = "កខច"
    encoded = converter.encode(text)

    print("Original text:", text)
    print("Encoded tensor:", encoded)
    print("Encoded list:", encoded.tolist())
    print("-" * 40)

    # ---- DECODE TEST ----
    decoded = converter.decode(encoded)

    print("Decoded text:", decoded)
    print("-" * 40)

    # ---- CTC BLANK TEST ----
    # Simulate model output with blanks (0)
    ctc_output = torch.tensor([0, encoded[0], 0, encoded[1], encoded[2], 0])

    print("CTC output (with blanks):", ctc_output.tolist())
    print("Decoded (CTC cleaned):", converter.decode(ctc_output))
