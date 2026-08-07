import torch

class TokenChannel:
    def __init__(self):
        """
        初始化傳輸通道。
        現階段依據實驗規劃，不模擬真實網路的封包遺失 (Packet Loss) 或頻寬限制，
        純粹作為一個透傳 (Pass-through) 的介面。
        """
        print("[Channel] 初始化 Token 傳輸通道 (純透傳模式)")

    def transmit(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        模擬傳輸過程。
        :param tokens: 來自 Client 端的 token tensor
        :return: Server 端接收到的 token tensor
        """
        # 現階段不對 token 進行任何破壞或延遲模擬
        # 如果未來階段需要模擬傳輸瓶頸，可以在這裡加入 delay 邏輯
        received_tokens = tokens.clone() 
        
        return received_tokens