//
// Created by TYTY on 2021-12-24 024.
//

#include "image_io.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <wincodec.h>
#include <windows.h>
#include <shlwapi.h>

#include <array>
#include <format>
#include <memory>
#include <iostream>
#include <charconv>

#define HR_CHECK(A) \
    do { \
        hr = (A); \
        if (FAILED(hr)) { \
            goto finish; \
        } \
    } while (0)

IWICImagingFactory *pFactory;

std::string format_hresult(HRESULT hr) {
  char buf[15] = "0000000";
  auto result = std::to_chars(buf + 7, buf + 15, uint32_t(hr), 16);
  return {result.ptr - 8, 8};
}

std::string init_image_io() {
  HRESULT hr;

  HR_CHECK(CoInitialize(nullptr));
  HR_CHECK(CoCreateInstance(CLSID_WICImagingFactory,
                            nullptr,
                            CLSCTX_INPROC_SERVER,
                            IID_IWICImagingFactory,
                            (LPVOID *) (&pFactory)));

  finish:
  if (FAILED(hr)) {
    return "WINAPI failure: " + format_hresult(hr);
  }

  return "";
}

static std::array<GUID, 21> alpha_formats{
    GUID_WICPixelFormat16bppBGRA5551,
    GUID_WICPixelFormat32bppBGRA,
    GUID_WICPixelFormat32bppPBGRA,
    GUID_WICPixelFormat32bppRGBA,
    GUID_WICPixelFormat32bppPRGBA,

    GUID_WICPixelFormat64bppRGBA,
    GUID_WICPixelFormat64bppBGRA,
    GUID_WICPixelFormat64bppPRGBA,
    GUID_WICPixelFormat64bppPBGRA,
    GUID_WICPixelFormat128bppRGBAFloat,

    GUID_WICPixelFormat128bppPRGBAFloat,
    GUID_WICPixelFormat64bppRGBAFixedPoint,
    GUID_WICPixelFormat64bppBGRAFixedPoint,
    GUID_WICPixelFormat128bppRGBAFixedPoint,
    GUID_WICPixelFormat128bppRGBFixedPoint,

    GUID_WICPixelFormat64bppRGBAHalf,
    GUID_WICPixelFormat64bppPRGBAHalf,
    GUID_WICPixelFormat32bppRGBA1010102,
    GUID_WICPixelFormat32bppRGBA1010102XR,
    GUID_WICPixelFormat32bppR10G10B10A2,

    GUID_WICPixelFormat32bppR10G10B10A2HDR10
};

static GUID desire_format_opaque = GUID_WICPixelFormat24bppBGR;
static GUID desire_format_alpha = GUID_WICPixelFormat32bppPBGRA;

std::variant<std::pair<shape_t<int32_t, 3>, mem_owner>, std::string>
load_image(Work::input_t file, bool ignore_alpha) {
  IStream *pStream = nullptr;
  IWICBitmapDecoder *pDecoder = nullptr;
  IWICBitmapFrameDecode *pFrame = nullptr;
  IWICFormatConverter *pConverter = nullptr;
  HRESULT hr;
  md_view<uint8_t, int32_t, 3> view;
  mem_owner pixels;

  if (file.index() == 0) {
    HR_CHECK(pFactory->CreateDecoderFromFilename(std::get<0>(file).c_str(),
                                                 nullptr,
                                                 GENERIC_READ,
                                                 WICDecodeMetadataCacheOnDemand,
                                                 &pDecoder));
  }
  else {
    auto pVec = std::get_if<1>(&file);
    if (pVec == nullptr) {
      return "unexpected input";
    }

    pStream = SHCreateMemStream(pVec->data(), pVec->size());
    HR_CHECK(pFactory->CreateDecoderFromStream(pStream, nullptr, WICDecodeMetadataCacheOnDemand, &pDecoder));
  }

  {
    HR_CHECK(pDecoder->GetFrame(0, &pFrame));

    UINT width, height;
    HR_CHECK(pFrame->GetSize(&width, &height));

    GUID input_format, desire_format;
    HR_CHECK(pFrame->GetPixelFormat(&input_format));
    bool has_alpha = std::find(alpha_formats.begin(), alpha_formats.end(), input_format) != alpha_formats.end();
    bool use_opaque = (ignore_alpha || !has_alpha);
    desire_format = use_opaque ? desire_format_opaque : desire_format_alpha;

    std::tie(view, pixels) = alloc_buffer<uint8_t>(height, width, use_opaque ? 3 : 4);
    if (input_format != desire_format) {
      HR_CHECK(pFactory->CreateFormatConverter(&pConverter));
      HR_CHECK(pConverter->Initialize(pFrame,
                                      desire_format,
                                      WICBitmapDitherTypeNone,
                                      nullptr,
                                      0.0,
                                      WICBitmapPaletteTypeCustom));

      HR_CHECK(pConverter->CopyPixels(nullptr,
                                      view.at(0).size(),
                                      view.size(),
                                      reinterpret_cast<BYTE *>(pixels.get())));
    }
    else {
      HR_CHECK(pFrame->CopyPixels(nullptr,
                                  view.at(0).size(),
                                  view.size(),
                                  reinterpret_cast<BYTE *>(pixels.get())));
    }
  }

  finish:
  pStream && pStream->Release();
  pConverter && pConverter->Release();
  pFrame && pFrame->Release();
  pDecoder && pDecoder->Release();

  if (FAILED(hr)) {
    return "WINAPI failure: " + format_hresult(hr);
  }

  return std::make_pair(view.shape, std::move(pixels));
}

std::string save_image(Work::output_t file, md_view<uint8_t, size_t, 3> data) {
  IWICStream *pStream = nullptr;
  IWICBitmapEncoder *pEncoder = nullptr;
  IWICBitmapFrameEncode *pFrame = nullptr;
  IWICBitmap *pSource = nullptr;
  HRESULT hr;

  if (file.index() != 0) {
    return "unimplemented";
  }

  {
    HR_CHECK(pFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &pEncoder));
    HR_CHECK(pFactory->CreateStream(&pStream));
    HR_CHECK(pStream->InitializeFromFilename(std::get<0>(file).c_str(), GENERIC_WRITE));
    HR_CHECK(pEncoder->Initialize(pStream, WICBitmapEncoderNoCache));
    HR_CHECK(pEncoder->CreateNewFrame(&pFrame, nullptr));
    HR_CHECK(pFrame->Initialize(nullptr));
    auto [height, width, components] = data.shape;
    auto output_format = components == 3 ? desire_format_opaque : desire_format_alpha;
    HR_CHECK(pFrame->SetPixelFormat(&output_format));
    HR_CHECK(pFrame->SetSize(width, height));

    HR_CHECK(pFactory->CreateBitmapFromMemory(width,
                                              height,
                                              output_format,
                                              data.at(0).size(),
                                              data.size(),
                                              reinterpret_cast<BYTE *>(data.data),
                                              &pSource));

    HR_CHECK(pFrame->WriteSource(pSource, nullptr));
    HR_CHECK(pFrame->Commit());
    HR_CHECK(pEncoder->Commit());
  }

  finish:
  pSource && pSource->Release();
  pFrame && pFrame->Release();
  pStream && pStream->Release();
  pEncoder && pEncoder->Release();

  if (FAILED(hr)) {
    return "WINAPI failure: " + format_hresult(hr);
  }

  return "";
}
