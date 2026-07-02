import Foundation
import MLX

public enum MLXImageProcessingError: LocalizedError {
    case invalidShape([Int])
    case unsupportedImageSource(String)

    public var errorDescription: String? {
        switch self {
        case .invalidShape(let shape):
            return "Expected an [H, W, 3] (or [1, H, W, 3] / planar [3, H, W]) image, got shape \(shape)"
        case .unsupportedImageSource(let message):
            return message
        }
    }
}

/// Still-image preprocessing implemented with pure MLX ops, so it runs on
/// every MLX backend (Metal, CUDA, CPU) — unlike ``MediaProcessing``, which
/// is CoreImage-based and only exists on Apple platforms. This is the path
/// that makes `UserInput.Image.array` usable on Linux (e.g. camera frames
/// on NVIDIA Jetson).
///
/// Fidelity note: resizing is bilinear (align-corners=false, matching
/// PyTorch's `mode="bilinear"`), whereas the Apple path uses bicubic.
/// For camera pipelines prefer delivering frames at the model's native
/// resolution (e.g. via GStreamer `videoscale`/`nvvidconv`) so this resize
/// is a no-op. Bicubic parity is a follow-up.
public enum MLXImageProcessing {

    /// Bilinear-resizes an `[H, W, C]` float image using pure MLX ops.
    public static func resizeBilinear(_ image: MLXArray, height outH: Int, width outW: Int)
        -> MLXArray
    {
        let srcH = image.dim(0)
        let srcW = image.dim(1)
        if srcH == outH && srcW == outW {
            return image
        }

        func axisCoordinates(_ out: Int, _ src: Int) -> (MLXArray, MLXArray, MLXArray) {
            let scale = Float(src) / Float(out)
            let x = (MLXArray(0 ..< out).asType(.float32) + 0.5) * scale - 0.5
            let x0 = clip(floor(x), min: 0, max: Float(src - 1))
            let x1 = clip(x0 + 1, min: 0, max: Float(src - 1))
            let fraction = clip(x - x0, min: 0, max: 1)
            return (x0.asType(.int32), x1.asType(.int32), fraction)
        }

        let (y0, y1, wy) = axisCoordinates(outH, srcH)
        let (x0, x1, wx) = axisCoordinates(outW, srcW)

        let top = image.take(y0, axis: 0)
        let bottom = image.take(y1, axis: 0)
        let topLeft = top.take(x0, axis: 1)
        let topRight = top.take(x1, axis: 1)
        let bottomLeft = bottom.take(x0, axis: 1)
        let bottomRight = bottom.take(x1, axis: 1)

        let wxRow = wx.reshaped([1, outW, 1])
        let wyColumn = wy.reshaped([outH, 1, 1])
        let topMix = topLeft * (1 - wxRow) + topRight * wxRow
        let bottomMix = bottomLeft * (1 - wxRow) + bottomRight * wxRow
        return topMix * (1 - wyColumn) + bottomMix * wyColumn
    }

    /// Prepares a raw RGB image for a SigLIP-style vision tower (Gemma 3):
    /// accepts `[H, W, 3]`, `[1, H, W, 3]`, planar `[3, H, W]`, or `[H, W, 4]`
    /// (alpha dropped), as uint8 0–255 or float 0–1, at any resolution.
    /// Returns `[1, 3, height, width]` float32, resized and normalized —
    /// the same output contract as the CoreImage pipeline
    /// (`resampleBicubic` + `normalize` + `asMLXArray`).
    ///
    /// The sRGB tone-curve step of the Apple path is intentionally absent:
    /// camera frames and decoded JPEGs are already sRGB-encoded, and on the
    /// Apple path linear-decode + re-encode cancel out for such input.
    public static func prepare(
        image: MLXArray,
        size: (height: Int, width: Int),
        mean: (Float, Float, Float),
        std: (Float, Float, Float)
    ) throws -> MLXArray {
        var array = image

        if array.ndim == 4, array.dim(0) == 1 {
            array = array.squeezed(axis: 0)
        }
        guard array.ndim == 3 else {
            throw MLXImageProcessingError.invalidShape(image.shape)
        }
        // planar [C, H, W] -> [H, W, C]
        if (array.dim(0) == 3 || array.dim(0) == 4), array.dim(2) != 3, array.dim(2) != 4 {
            array = array.transposed(1, 2, 0)
        }
        switch array.dim(2) {
        case 3:
            break
        case 4:
            array = array[0..., 0..., ..<3]
        default:
            throw MLXImageProcessingError.invalidShape(image.shape)
        }

        array = array.asType(.float32)
        if array.max().item(Float.self) > 1.0 {
            array = array / 255
        }

        array = resizeBilinear(array, height: size.height, width: size.width)

        let meanArray = MLXArray([mean.0, mean.1, mean.2]).reshaped([1, 1, 3])
        let stdArray = MLXArray([std.0, std.1, std.2]).reshaped([1, 1, 3])
        array = (array - meanArray) / stdArray

        // [H, W, C] -> [1, C, H, W]
        return expandedDimensions(array.transposed(2, 0, 1), axis: 0)
    }
}
