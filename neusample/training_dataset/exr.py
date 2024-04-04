import OpenEXR, Imath, numpy

def info(filename):
    img = OpenEXR.InputFile(filename)
    for k, v in list(img.header().items()):
        print((k, v))

def get_size(filename):
    img = OpenEXR.InputFile(filename)
    header = img.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    return size

def read(filename):
    img = OpenEXR.InputFile(filename)
    header = img.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    def chan(c):
        s = img.channel(c, pt)
        arr = numpy.fromstring(s, dtype=numpy.float32)
        arr.shape = size[1], size[0]
        return arr

    # single-channel file
    channels = list(header['channels'].keys())
    if len(channels) == 1:
        return chan(channels[0])
    elif len(channels) == 3:
        return numpy.dstack([chan('R'), chan('G'), chan('B')])
    elif len(channels) == 4:
        return numpy.dstack([chan('R'), chan('G'), chan('B'), chan('A')])
    else:
        assert False


def write32(img, filename):
    assert img.dtype == numpy.float32
    h, w, d = img.shape
    assert d == 3

    header = OpenEXR.Header(w, h)
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    out = OpenEXR.OutputFile(filename, header)

    r = img[:,:,0].tostring()
    g = img[:,:,1].tostring()
    b = img[:,:,2].tostring()
    out.writePixels({'R': r, 'G': g, 'B': b})


def writeMono32(img, filename):
    assert img.dtype == numpy.float32
    assert len(img.shape) == 2
    h, w = img.shape

    header = OpenEXR.Header(w, h)
    out = OpenEXR.OutputFile(filename, header)

    data = img.tostring()
    out.writePixels({'R': data, 'G': data, 'B': data})


def write16(img, filename):
    assert img.dtype == numpy.float32
    h, w, d = img.shape
    assert d == 3

    header = OpenEXR.Header(w, h)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))

    header['channels'] = dict([(c, half_chan) for c in "RGB"])
    out = OpenEXR.OutputFile(filename, header)
    r = img[:,:,0].astype(numpy.float16).tostring()
    g = img[:,:,1].astype(numpy.float16).tostring()
    b = img[:,:,2].astype(numpy.float16).tostring()
    out.writePixels({'R': r, 'G': g, 'B': b})


def writeMono16(img, filename):
    assert img.dtype == numpy.float32
    assert len(img.shape) == 2
    h, w = img.shape

    header = OpenEXR.Header(w, h)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))

    header['channels'] = dict([(c, half_chan) for c in "RGB"])
    out = OpenEXR.OutputFile(filename, header)
    data = img.astype(numpy.float16).tostring()
    out.writePixels({'R': data, 'G': data, 'B': data})

