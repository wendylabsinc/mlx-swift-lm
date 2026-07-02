import MLXLMCommon

// Extracted from Qwen2VL.swift: platform-neutral (no CoreImage), and also
// used by models that are available on Linux (Gemma3).

/// Message Generator for Qwen2VL
public struct Qwen2VLMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        [
            "role": message.role.rawValue,
            "content": [
                ["type": "text", "text": message.content]
            ]
                // Messages format for Qwen 2 VL, Qwen 2.5 VL. May need to be adapted for other models.
                + message.images.map { _ in
                    ["type": "image"]
                }
                + message.videos.map { _ in
                    ["type": "video"]
                },
        ]
    }
}
