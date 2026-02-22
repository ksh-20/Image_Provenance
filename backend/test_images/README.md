# Test Images

Place test images here to try the deepfake detector.

## Recommended Test Images

1. **Real Photos**:

   - Photos from your phone camera
   - Professional photographs
   - Screenshots

2. **AI-Generated Images**:

   - Images from DALL-E, Midjourney, Stable Diffusion
   - StyleGAN generated faces
   - Face swap apps

3. **Manipulated Images**:
   - Photoshopped images
   - Heavily filtered photos
   - Face-swapped images

## How to Test

```bash
# Run the test script
python test_free_api.py test_images/your_image.jpg
```

Or upload directly through the web interface:
`http://localhost:3000/deepfake-check`

## Expected Results

- **Real Photos**: Should show "likely authentic" (< 30% probability)
- **AI-Generated**: Should show "likely deepfake" (> 60% probability)
- **Edited Photos**: May show "possibly manipulated" (30-60% probability)

Remember: Results are for educational purposes and not definitive proof!
