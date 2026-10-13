import torch
from models.light_sakt import LightSAKT

def test_model():
    print('Testing LightSAKT model instantiation...')
    model = LightSAKT(num_q=100, n=50, d=128, num_attn_heads=8, dropout=0.1)
    print(f'Model created successfully with {sum(p.numel() for p in model.parameters())} parameters')

    # Test forward pass
    batch_size = 4
    seq_len = 50
    q = torch.randint(0, 100, (batch_size, seq_len))
    r = torch.randint(0, 2, (batch_size, seq_len))
    qry = torch.randint(0, 100, (batch_size, seq_len))

    print('Testing forward pass...')
    with torch.no_grad():
        output, _ = model(q, r, qry)
        print(f'Forward pass successful. Output shape: {output.shape}')
        print(f'Output range: [{output.min():.4f}, {output.max():.4f}]')
    
    # Test linear attention mechanism
    print('\nTesting linear attention mechanism...')
    from models.light_sakt import LinearAttention
    attn = LinearAttention(d_model=128, n_heads=8)
    q = torch.randn(50, batch_size, 128)  # seq_len, batch_size, d_model
    k = torch.randn(50, batch_size, 128)
    v = torch.randn(50, batch_size, 128)
    mask = torch.triu(torch.ones(50, 50), diagonal=1).bool()
    
    output, _ = attn(q, k, v, mask)
    print(f'Linear attention output shape: {output.shape}')
    print(f'Linear attention memory usage: {torch.cuda.max_memory_allocated() if torch.cuda.is_available() else "N/A"} bytes')

    print('\nBasic model test completed successfully!')

if __name__ == "__main__":
    test_model()
