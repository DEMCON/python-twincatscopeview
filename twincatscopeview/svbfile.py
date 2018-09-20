'''
reader for TwinCAT SVB files

reference: https://infosys.beckhoff.com/english.php?content=../content/1033/te13xx_tc3_scopeview/36028797201259531.html

'''

import struct
import numpy as np
import datetime
import mmap
import collections.abc
import io

def _readheader(file, format):
    '''
    helper function to read a header from a file stream
    
    format is similar to struct.pack/unpack format, with these differences:
    
    - The * format char denotes a utf-8 encoded string preceded by a uint32 length
    - The T format char denotes a datetime encoded as a uint64 timestamp, 100ns
        units since 1-1-1601
    - The endinaness is always little endian
    - Repetition counts are not supported
    '''
    
    fields = []
    for f in format:
        if f == '*': # string array, coded as Int32 length + utf-8 data
            strlen, = struct.unpack('<L', file.read(4))
            fields.append(file.read(strlen).decode('utf-8'))
        elif f == 'T': # datetime, coded as 100ns periods since 1-1-1601
            timeval, = struct.unpack('<Q', file.read(8))
            
            fields.append(
                datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds = timeval/10)
                )
        else:
            fieldlength = struct.calcsize('<' + f)
            fields.append(struct.unpack('<' + f, file.read(fieldlength))[0])

    return fields


class Channel:
    DATATYPES = {
        'REAL64'  : np.float64,
        'REAL32'  : np.float32,
        'UINT64'  : np.uint64,
        'UINT32'  : np.uint32,
        'UINT16'  : np.uint16,
        'UINT8'   : np.uint8,
        'INT64'   : np.int64,
        'INT32'   : np.int32,
        'INT16'   : np.int16,
        'INT8'    : np.byte,
        'BIT'     : np.bool,
        'BIT8'    : np.bool8,
        'BITARR8' : np.int8,
        'BITARR16': np.int16,
        'BITARR32': np.int32
        }
        
    def __init__(self, headerfile, file, starttime):
        '''
        Channel represents a single data channel in an SVBFile
        
        
        headerfile: file object from which to read the header record
        file: file object in which the channel data is stored
        starttime: datetime.datetime object of the start of the data
 
         
        note: the np.memmap constructor, which is called from here, rewinds
        the provided file object. Therefore, headerfile and file should not be
        the same file object (otherwise headerfile would be rewound too)
        '''
        # self._time and self._datetime store the cached results of the Time and
        # Datetime properties
        self._time = None
        self._datetime = None
        self.StartTime = starttime # required to generate Datetime property

        startpos = headerfile.tell()
        
        # There is an error in the documentation: fields Offset and Scalefactor 
        # are swapped. The next line works correctly.
        
        headersize, self.Name, self.NetId, self.Port, sampleTime, self.SymbolBased, \
            self.SymbolName, self.Comment, self.IndexGroup, self.IndexOffset, self.DataType, \
            self.DataTypeId, self.VariableSize, self.SamplesInFile, self.DataInFile, \
            self.FileStartPosition, self.Offset, self.Scalefactor, self.Bitmask \
                = _readheader(headerfile, 'Q**LQ?**QQ*LLQQQddQ')
        
        # Check headersize
        if headerfile.tell() - startpos != headersize:
            raise IOError('Error reading file: channel header size mismatch')
        
        self.SampleTime = sampleTime * 100e-9 # Convert 100ns units to seconds
        
        assert self.DataInFile == (self.VariableSize + 4) * self.SamplesInFile
        
        self.Data = np.memmap(file, [('Timestamp', np.uint32), ('Values', self.DATATYPES[self.DataType])], 'r', self.FileStartPosition, (self.SamplesInFile,))
        
        # numpy array (uint32) with the timestamp as stored in the file [100ns units]
        # Be aware that the timestamps wraps after 100*2**32ns = 429.5s
        self.Timestamp = self.Data['Timestamp']
        
        # numpy array with the data values as stored in the file
        self.Values = self.Data['Values'] 

    # Time and Datetime are only generated when requested for the first time
    # to improve loading speed
    @property
    def Time(self):
        '''
        Returns a numpy array (float64) with the timestamp in seconds, undoing any
        wrapping that may occur
        '''
        if self._time is None:
            
            t = self.Timestamp
            
            # find dt's - takes into account integer overflow
            dt = np.diff(t)
    
            # create new time array
            result = np.empty(t.shape, np.float64)
            
            # fill time array
            result[0] = t[0]
            np.cumsum(dt, out = result[1:])
    
            result /= 1e7   # scale [100 ns] to [seconds]
            
            self._time = result
            
        return self._time

    @property
    def Datetime(self):
        '''
        Returns a list of Python datetime.datetime objects, one for each sample
        '''
        if self._datetime is None:
            self._datetime = [ self.StartTime + datetime.timedelta(seconds = s) for s in self.Time ]
            
        return self._datetime

    def interpolate(self, t):
        '''
        Interpolation of a data channel, useful to do processing on different
        channels with time arrays that are not equal.

        t: time values in seconds at which the data should be interpolated
 
        Returns: numpy array
        '''

        if np.all(t == self.Time): 
            # Short-cut if time axis is the same
            return np.array(self.Values)
        
        return np.interp(t, self.Time, self.Values)
        
    def __repr__(self):
        return '[%s]*%d @ %0.2f ms: %.25s' % (self.DataType, self.SamplesInFile, self.SampleTime*1000, self.Name)


class SVBFile(collections.abc.Mapping):
    def __init__(self, filename):
        self._channels = {}
        
        # Read the file into a object
        file = open(filename, mode='rb')
        headerSize, = struct.unpack('<Q', file.read(8))
        header = io.BytesIO(file.read(headerSize - 8))
        
        self.Name, self.StartTime, self.EndTime, self.ChannelCount = _readheader(header, '*TTL')
        
        #Channel Header
        for i in range(self.ChannelCount):
            ch = Channel(header, file, self.StartTime)
            self._channels[ch.Name] = ch
    
        # Check all header bytes processed
        if header.read(1):
            raise IOError('Error reading file: file header size mismatch')
                                    
    def __repr__(self):
        return '\n'.join(repr(ch) for ch in self._channels.values())
   
    # Mapping protocol
    def __getitem__(self, key):
        return self._channels[key]
    
    def __iter__(self):
        return iter(self._channels)
    
    def __len__(self):
        return len(self._channels)