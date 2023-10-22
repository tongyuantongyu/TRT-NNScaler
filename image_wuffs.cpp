#define WUFFS_IMPLEMENTATION

#define WUFFS_CONFIG__STATIC_FUNCTIONS

#define WUFFS_CONFIG__MODULES
#define WUFFS_CONFIG__MODULE__ADLER32
#define WUFFS_CONFIG__MODULE__AUX__BASE
#define WUFFS_CONFIG__MODULE__AUX__IMAGE
#define WUFFS_CONFIG__MODULE__BASE
#define WUFFS_CONFIG__MODULE__BMP
#define WUFFS_CONFIG__MODULE__CRC32
#define WUFFS_CONFIG__MODULE__DEFLATE
#define WUFFS_CONFIG__MODULE__GIF
#define WUFFS_CONFIG__MODULE__LZW
#define WUFFS_CONFIG__MODULE__NIE
#define WUFFS_CONFIG__MODULE__PNG
#define WUFFS_CONFIG__MODULE__TGA
#define WUFFS_CONFIG__MODULE__WBMP
#define WUFFS_CONFIG__MODULE__ZLIB

#include "wuffs-v0.3.c"

#include <cstdio>
#include <cmath>
#include <fstream>
#include <optional>
#include <vector>

#include "jpeglib.h"
#include <setjmp.h>

#include "image_io.h"

class MyDecodeImageCallbacks : public wuffs_aux::DecodeImageCallbacks {
 public:
  MyDecodeImageCallbacks(bool ignore_alpha) : m_combined_gamma(1.0), ignore_alpha{ignore_alpha} {}

 private:
  wuffs_base__pixel_format SelectPixfmt(const wuffs_base__image_config &image_config) override {
    if (ignore_alpha || image_config.first_frame_is_opaque()) {
      return wuffs_base__make_pixel_format(WUFFS_BASE__PIXEL_FORMAT__BGR);
    }
    else {
      return wuffs_base__make_pixel_format(WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL);
    }
  }

  AllocPixbufResult AllocPixbuf(const wuffs_base__image_config &image_config,
                                bool) override {
    uint32_t w = image_config.pixcfg.width();
    uint32_t h = image_config.pixcfg.height();
    if ((w == 0) || (h == 0)) {
      return {""};
    }
    uint64_t len = image_config.pixcfg.pixbuf_len();
    if ((len == 0) || (SIZE_MAX < len)) {
      return {wuffs_aux::DecodeImage_UnsupportedPixelConfiguration};
    }
    auto mem = std::make_unique<uint8_t[]>(len);
    if (!mem) {
      return {wuffs_aux::DecodeImage_OutOfMemory};
    }
    wuffs_base__pixel_buffer pixbuf;
    wuffs_base__status status = pixbuf.set_from_slice(
        &image_config.pixcfg,
        wuffs_base__make_slice_u8(mem.get(), (size_t) len));
    if (!status.is_ok()) {
      return {status.message()};
    }
    wuffs_aux::MemOwner owner {mem.release(), operator delete[]};
    return {std::move(owner), pixbuf};
  }

  std::string  //
  HandleMetadata(const wuffs_base__more_information &minfo,
                 wuffs_base__slice_u8 raw) override {
    if (minfo.flavor == WUFFS_BASE__MORE_INFORMATION__FLAVOR__METADATA_PARSED) {
      switch (minfo.metadata__fourcc()) {
        case WUFFS_BASE__FOURCC__GAMA:
          // metadata_parsed__gama returns the inverse gamma scaled by 1e5.
          m_combined_gamma =
              1e5 / (2.2 * minfo.metadata_parsed__gama());
          break;
      }
    }
    return wuffs_aux::DecodeImageCallbacks::HandleMetadata(minfo, raw);
  }

  void  //
  Done(wuffs_aux::DecodeImageResult &result,
       wuffs_aux::sync_io::Input &input,
       wuffs_aux::IOBuffer &buffer,
       wuffs_base__image_decoder::unique_ptr image_decoder) override {
    if ((result.pixbuf.pixel_format().repr ==
        WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL) &&
        ((m_combined_gamma < 0.9999) || (1.0001 < m_combined_gamma))) {
      uint8_t lut[256];
      lut[0x00] = 0x00;
      lut[0xFF] = 0xFF;
      for (uint32_t i = 1; i < 0xFF; i++) {
        lut[i] =
            (uint8_t) (floor(255.0 * pow(i / 255.0, m_combined_gamma) + 0.5));
      }

      wuffs_base__table_u8 t = result.pixbuf.plane(0);
      size_t w4 = t.width / 4;
      for (size_t y = 0; y < t.height; y++) {
        uint8_t *ptr = t.ptr + (y * t.stride);
        for (size_t x = 0; x < w4; x++) {
          ptr[0] = lut[ptr[0]];
          ptr[1] = lut[ptr[1]];
          ptr[2] = lut[ptr[2]];
          ptr += 4;
        }
      }
    }
  }

  // m_combined_gamma holds the product of the screen gamma and the image
  // file's inverse gamma.
  double m_combined_gamma;
  bool ignore_alpha;
};

std::string init_image_io() {
  return "";
}

struct my_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
  std::string err_info;
};

METHODDEF(void)
my_error_exit (j_common_ptr cinfo)
{
  my_error_mgr * myerr = (my_error_mgr *) cinfo->err;
  myerr->err_info.resize(JMSG_LENGTH_MAX);
  myerr->pub.format_message(cinfo, myerr->err_info.data());
  longjmp(myerr->setjmp_buffer, 1);
}

std::optional<std::pair<shape_t<3>, mem_owner>>
load_image_jpeg(std::variant<FILE *, std::vector<uint8_t>> f) {
  struct jpeg_decompress_struct cinfo;
  struct my_error_mgr jerr;
  JSAMPARRAY buffer;
  int row_stride;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = my_error_exit;
  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_decompress(&cinfo);
    return {};
  }

  jpeg_create_decompress(&cinfo);
  if (f.index() == 0) {
    jpeg_stdio_src(&cinfo, std::get<0>(f));
  }
  else {
    auto &vec = std::get<1>(f);
    jpeg_mem_src(&cinfo, vec.data(), vec.size());
  }

  (void) jpeg_read_header(&cinfo, TRUE);
  cinfo.out_color_space = JCS_EXT_BGR;
  (void) jpeg_start_decompress(&cinfo);

  row_stride = cinfo.output_width * cinfo.output_components;
  auto [in_view, in_ptr] = alloc_buffer<uint8_t>(cinfo.output_height, cinfo.output_width, cinfo.output_components);
  buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
  int n = 0;
  while (cinfo.output_scanline < cinfo.output_height) {
    (void) jpeg_read_scanlines(&cinfo, buffer, 1);
    memcpy(in_view.at(n).data, buffer[0], row_stride);
    ++n;
  }

  (void) jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return std::make_pair(in_view.shape, std::move(in_ptr));
}

std::variant<std::pair<shape_t<3>, mem_owner>, std::string>
load_image(Work::input_t file, bool ignore_alpha) {
  wuffs_aux::DecodeImageResult res("");
  if (file.index() == 0) {
    auto name = std::get<0>(file);
  FILE *f = nullptr;
#ifdef _WIN32
    _wfopen_s(&f, name.c_str(), L"rb");
#else
    f = fopen64(name.c_str(), "rb");
#endif
  if (f == nullptr) {
    return "can't open file";
  }

  wuffs_aux::sync_io::FileInput input(f);

    MyDecodeImageCallbacks callbacks(ignore_alpha);
    res = wuffs_aux::DecodeImage(
      callbacks,
      input,
      wuffs_aux::DecodeImageArgQuirks::DefaultValue(),
      wuffs_aux::DecodeImageArgFlags(wuffs_aux::DecodeImageArgFlags::REPORT_METADATA_GAMA));

  if (!res.error_message.empty()) {
    if (res.error_message == wuffs_aux::DecodeImage_UnsupportedImageFormat) {
      fseek(f, 0, SEEK_SET);
      auto result = load_image_jpeg(f);
      if (result) {
        return std::move(*result);
      }
    }

    fclose(f);
    return "failed decoding image: " + res.error_message;
  }

    fclose(f);
  }
  else {
    auto pVec = std::get_if<1>(&file);
    if (pVec == nullptr) {
      return "unexpected input";
    }

    wuffs_aux::sync_io::MemoryInput input(pVec->data(), pVec->size());

    MyDecodeImageCallbacks callbacks(ignore_alpha);
    res = wuffs_aux::DecodeImage(
        callbacks,
        input,
        wuffs_aux::DecodeImageArgQuirks::DefaultValue(),
        wuffs_aux::DecodeImageArgFlags(wuffs_aux::DecodeImageArgFlags::REPORT_METADATA_GAMA));

    if (!res.error_message.empty()) {
      if (res.error_message == wuffs_aux::DecodeImage_UnsupportedImageFormat) {
        auto result = load_image_jpeg(std::move(*pVec));
        if (result) {
          return std::move(*result);
        }
      }

    return "failed decoding image: " + res.error_message;
    }
  }

  md_view in_view{reinterpret_cast<uint8_t *>(res.pixbuf_mem_owner.get()),
                  {res.pixbuf.pixcfg.height(),
                   res.pixbuf.pixcfg.width(),
                   res.pixbuf.pixcfg.pixel_format().transparency() ? 4 : 3}};

  std::unique_ptr<uint8_t[]> in_ptr(reinterpret_cast<uint8_t*>(res.pixbuf_mem_owner.release()));
  return std::make_pair(in_view.shape, std::move(in_ptr));
}
