#pragma once

#ifdef PARTICLELIFE_EMBED_RESOURCES

#include <cstddef>
#include <string>
#include <string_view>

namespace EmbeddedResources {

    // Shaders
    inline unsigned char particleVert[] = {
#include "particle.vert.h"
    };

    inline unsigned char particleFrag[] = {
#include "particle.frag.h"
    };

    inline unsigned char particleCompositeVert[] = {
#include "particle_composite.vert.h"
    };

    inline unsigned char particleCompositeFrag[] = {
#include "particle_composite.frag.h"
    };

    inline unsigned char particleGlowExtractFrag[] = {
#include "particle_glow_extract.frag.h"
    };

    inline unsigned char particleGlowBlurFrag[] = {
#include "particle_glow_blur.frag.h"
    };

    inline unsigned char simulationComp[] = {
#include "simulation.comp.h"
    };

    // Fonts
    inline unsigned char jetBrainsMonoRegular[] = {
#include "JetBrainsMono-Regular.ttf.h"
    };

    inline unsigned char jetBrainsMonoSemiBold[] = {
#include "JetBrainsMono-SemiBold.ttf.h"
    };

    // Helpers
    struct DataView {
        unsigned char* data = nullptr;
        std::size_t    size = 0;

        explicit operator bool() const noexcept {
            return data != nullptr;
        }
    };

    inline DataView find_text(std::string_view path) noexcept {
        if (path == "resources/shaders/particle.vert")
            return {particleVert, sizeof(particleVert)};

        if (path == "resources/shaders/particle.frag")
            return {particleFrag, sizeof(particleFrag)};

        if (path == "resources/shaders/particle_composite.vert")
            return {
                particleCompositeVert,
                sizeof(particleCompositeVert)};

        if (path == "resources/shaders/particle_composite.frag")
            return {
                particleCompositeFrag,
                sizeof(particleCompositeFrag)};

        if (path == "resources/shaders/particle_glow_extract.frag")
            return {
                particleGlowExtractFrag,
                sizeof(particleGlowExtractFrag)};

        if (path == "resources/shaders/particle_glow_blur.frag")
            return {
                particleGlowBlurFrag,
                sizeof(particleGlowBlurFrag)};

        if (path == "resources/shaders/simulation.comp")
            return {
                simulationComp,
                sizeof(simulationComp)};

        return {};
    }

    inline std::string load_text(std::string_view path) {
        const DataView view = find_text(path);

        if (!view)
            return {};

        return std::string(
            reinterpret_cast<const char*>(view.data),
            view.size);
    }

} // namespace EmbeddedResources

#endif
