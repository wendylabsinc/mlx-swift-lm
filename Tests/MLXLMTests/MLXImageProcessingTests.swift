import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

#if canImport(CoreImage)
    import CoreImage
#endif

@Suite("MLXImageProcessing")
struct MLXImageProcessingTests {

    init() {
        // The CUDA fork of mlx-swift does not bundle the metallib for SPM
        // test runs; the CPU backend verifies the same numerics.
        Device.setDefault(device: .cpu)
    }

    @Test("resize to same size is identity")
    func resizeIdentity() throws {
        let image = MLXArray.ones([8, 8, 3]).asType(.float32) * 0.25
        let out = MLXImageProcessing.resizeBilinear(image, height: 8, width: 8)
        #expect(abs(out - image).max().item(Float.self) < 1e-6)
    }

    @Test("2x upscale interpolates midpoints")
    func upscale() throws {
        // one column black, one white -> midpoints at 0.25 / 0.75 with
        // align-corners=false bilinear
        let image = MLXArray([Float]([0, 1, 0, 1]), [2, 2, 1])
        let out = MLXImageProcessing.resizeBilinear(image, height: 2, width: 4)
        let row: [Float] = out[0, 0..., 0].asArray(Float.self)
        #expect(row.count == 4)
        #expect(Swift.abs(row[0] - 0.0) < 1e-5)
        #expect(Swift.abs(row[1] - 0.25) < 1e-5)
        #expect(Swift.abs(row[2] - 0.75) < 1e-5)
        #expect(Swift.abs(row[3] - 1.0) < 1e-5)
    }

    @Test("prepare produces normalized NCHW float32")
    func prepareContract() throws {
        // solid gray 128 uint8, arbitrary input size
        let gray = (MLXArray.ones([100, 160, 3]) * 128).asType(.uint8)
        let out = try MLXImageProcessing.prepare(
            image: gray, size: (height: 896, width: 896),
            mean: (0.5, 0.5, 0.5), std: (0.5, 0.5, 0.5))

        #expect(out.shape == [1, 3, 896, 896])
        #expect(out.dtype == .float32)
        // ((128/255) - 0.5) / 0.5 = 0.00392...
        let value = out.mean().item(Float.self)
        #expect(Swift.abs(value - 0.00392) < 1e-3)
    }

    #if canImport(CoreImage)
        @Test("parity with the CoreImage pipeline at native resolution")
        func parityWithCoreImage() throws {
            // Deterministic gradient image at the pipeline's target size so
            // the resize step is a no-op on both paths — isolates tone curve
            // + normalize behavior.
            let size = 64
            let ramp = MLXArray(0 ..< size).asType(.float32) / Float(size - 1)
            let horizontal = MLXArray.ones([size, 1]) * ramp.reshaped([1, size])
            let vertical = ramp.reshaped([size, 1]) * MLXArray.ones([1, size])
            let constant = MLXArray.ones([size, size]) * 0.5
            let image = stacked([horizontal, vertical, constant], axis: -1)

            // MLX path
            let mlxOut = try MLXImageProcessing.prepare(
                image: image, size: (height: size, width: size),
                mean: (0.5, 0.5, 0.5), std: (0.5, 0.5, 0.5))

            // CoreImage path, mirroring Gemma3Processor.preprocess(images:)
            let uint8Image = (image * 255).asType(.uint8)
            let ciImage = try UserInput.Image.array(uint8Image).asCIImage()
            let srgb = MediaProcessing.inSRGBToneCurveSpace(ciImage)
            let resized = MediaProcessing.resampleBicubic(
                srgb, to: CGSize(width: size, height: size))
            let normalized = MediaProcessing.normalize(
                resized, mean: (0.5, 0.5, 0.5), std: (0.5, 0.5, 0.5))
            let ciOut = MediaProcessing.asMLXArray(normalized)

            #expect(ciOut.shape == mlxOut.shape)
            let maxDiff = abs(ciOut - mlxOut).max().item(Float.self)
            // uint8 quantization alone accounts for ~0.008 in normalized
            // units; a tone-curve mismatch would show up as > 0.1.
            #expect(maxDiff < 0.05, "max abs difference: \(maxDiff)")
        }
    #endif
}
