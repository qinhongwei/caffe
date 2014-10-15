"""Monitors a log directory and provides a site for visualizing the logs."""

# pylint: disable=too-many-public-methods
# pylint: disable=bad-builtin

from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.web import RequestHandler, asynchronous, Application, \
    StaticFileHandler
from tornado import gen
from tornado.options import define, options, parse_command_line

from glob import glob
import json
import struct
from itertools import chain

from caffe.proto.caffe_pb2 import LogRecord, VisualizationRecord, \
    InspectionRecord, LossRecord, ScoreRecord

define('log_directory', default='.')
define('port', default=80, type=int)
define('static_directory', default='static')

# pylint: disable=invalid-name
# pylint: disable=no-member
VisualizationRecordType = LogRecord.Type.Value('VISUALIZATION')
InspectionRecordType = LogRecord.Type.Value('INSPECTION')
LossRecordType = LogRecord.Type.Value('LOSS')
ScoreRecordType = LogRecord.Type.Value('SCORE')
# pylint: enable=invalid-name
# pylint: enable=no-member

def downsampler(iterable, max_count):
  """Resamples an iterable to produce a max_count number of elements."""

  num_records = len(iterable)

  skip = num_records / max_count + 1

  for pos, item in enumerate(iterable):
    if pos % skip == 0:
      yield item

class Log(object):
  """Manages reading a directory of log files."""

  def __init__(self, log_directory):
    """Initializes Log object."""

    self._logs = {}
    self._records = {
      VisualizationRecordType: None,
      InspectionRecordType: [],
      LossRecordType: [],
      ScoreRecordType: [],
    }
    self._log_directory = log_directory

  def _update_files(self):
    """Updates the list of log files in the log directory."""

    logs = glob('%s/*.log' % self._log_directory)

    for log in logs:
      if log not in self._logs:
        self._logs[log] = open(log, 'rb')

  def _sort_records(self):
    """Sorts the records by timestamp."""

    for record_type in self._records.values():
      if type(record_type) is list:
        record_type.sort(key=lambda k: k.timestamp)

  def _update_records(self):
    """Checks the log files for new records."""
    self._update_files()

    for log in self._logs.values():
      while True:
        print log.tell()
        length_bytes = log.read(8)
        if len(length_bytes) < 8:
          log.seek(-len(length_bytes), 1)
          break

        length, = struct.unpack('Q', length_bytes)
        print 'length %d' % length
        serialized = log.read(length)

        if len(serialized) < length:
          log.seek(-len(serialized), 1)
          log.seek(-8, 1)
          break
        assert len(serialized) == length

        record = LogRecord()
        record.ParseFromString(serialized)

        # pylint: disable=no-member
        record_type = record.type
        # pylint: enable=no-member

        if type(self._records[record_type]) is list:
          self._records[record_type].append(record)
        else:
          self._records[record_type] = record

    self._sort_records()

  def get_visualization_record(self):
    """Get the most recent visualization record."""

    return self._records[VisualizationRecordType]

  def get_inspection_records(self):
    """Get all of the inspection records."""

    return self._records[InspectionRecordType]

  def get_loss_records(self):
    """Get all of the loss records."""

    return self._records[LossRecordType]

  def get_score_records(self):
    """Get all of the score records."""

    return self._records[ScoreRecordType]

def wrapper(fcn, callback=None):
  callback(fcn())

class VisualizationHandler(RequestHandler):
  """Visualization endpoint handler."""

  @asynchronous
  @gen.engine
  # pylint: disable=arguments-differ
  def get(self, mode, name):
  # pylint: enable=arguments-differ
    """Respond to GET requests for the visualization endpoint."""

    viz_data = yield gen.Task(wrapper, __log__.get_visualization_record)

    response = yield gen.Task(self._process, viz_data, mode, name)

    self.set_header('content-type', 'image/png')
    self.write(response)
    self.finish()

  def _process(self, record, mode, name, callback):
    """GET callback."""

    # pylint: disable=no-member
    visualization_record = record.Extensions[VisualizationRecord.parent]
    # pylint: enable=no-member

    snapshots = None
    if mode == 'activations':
      snapshots = visualization_record.activation_snapshots
    else:
      snapshots = visualization_record.weight_snapshots

    name_tokens = name.split('-')
    layer_name = '-'.join(name_tokens[:-1])
    layer_index = name_tokens[-1]

    selected = None
    for entry in snapshots:
      if (entry.layer_name == layer_name and
          entry.layer_index == int(layer_index)):
        selected = entry
        break

    callback(selected.png)

class InspectionHandler(RequestHandler):
  """Inspection endpoint handler."""

  @asynchronous
  @gen.engine
  def get(self):
    """Respond to GET requests for inspection endpoint."""

    inspection_data = yield gen.Task(wrapper, __log__.get_inspection_records)

    response = yield gen.Task(self._process, inspection_data)

    self.set_header('content-type', 'application/json')
    self.write(response)
    self.finish()

  def _process(self, result, callback):
    """GET callback."""

    def record_formatter(record):
      """Reformat the record as a dict."""

      # pylint: disable=no-member
      inspection_record = record.Extensions[InspectionRecord.parent]
      # pylint: enable=no-member

      def result_formatter(result):
        """Reformat a single result as a dict."""

        return {
          'name': result.layer_name,
          'idx': result.layer_index,
          'w0': result.weight_0pct,
          'w15': result.weight_15pct,
          'w50': result.weight_50pct,
          'w85': result.weight_85pct,
          'w100': result.weight_100pct,
          'g0': result.grad_0pct,
          'g15': result.grad_15pct,
          'g50': result.grad_50pct,
          'g85': result.grad_85pct,
          'g100': result.grad_100pct,
          'ts': record.timestamp,
          'iter': record.iteration,
        }

      return map(result_formatter, inspection_record.inspection_results)

    formatted_records = map(record_formatter, downsampler(result, 40))
    formatted_records = [item for sublist in formatted_records
        for item in sublist]

    callback(json.dumps(formatted_records))

class LossHandler(RequestHandler):
  """Loss endpoint handler."""

  @asynchronous
  @gen.engine
  def get(self):
    """Respond to GET requests for loss endpoint."""

    loss_data = yield gen.Task(wrapper, __log__.get_loss_records)

    response = yield gen.Task(self._process, loss_data)

    self.set_header('content-type', 'application/json')
    self.write(response)
    self.finish()

  def _process(self, result, callback):
    """GET callback."""

    def record_formatter(record):
      """Reformat the record as a dict."""

      # pylint: disable=no-member
      loss_record = record.Extensions[LossRecord.parent]
      # pylint: enable=no-member

      return {
        'name': loss_record.name,
        'ts': record.timestamp,
        'iter': record.iteration,
        'loss': list(loss_record.loss),
      }

    formatted_records = map(record_formatter, result)

    separated = {}
    for item in formatted_records:
      if item['name'] not in separated:
        separated[item['name']] = []
      separated[item['name']].append(item)

    for name in separated:
      separated[name] = downsampler(separated[name], 100)

    callback(json.dumps(list(chain(*separated.values()))))

class ScoreHandler(RequestHandler):
  """Score endpoint handler."""

  @asynchronous
  @gen.engine
  def get(self):
    """Respond to GET requests for loss endpoint."""

    score_data = yield gen.Task(wrapper, __log__.get_score_records)

    response = yield gen.Task(self._process, score_data)

    self.set_header('content-type', 'application/json')
    self.write(response)
    self.finish()

  def _process(self, result, callback):
    """GET callback."""

    def record_formatter(record):
      """Reformat the record as a dict."""

      # pylint: disable=no-member
      score_record = record.Extensions[ScoreRecord.parent]
      # pylint: enable=no-member

      return {
        'name': score_record.name,
        'ts': record.timestamp,
        'iter': record.iteration,
        'score': score_record.score,
      }

    formatted_records = map(record_formatter, result)

    separated = {}
    for item in formatted_records:
      if item['name'] not in separated:
        separated[item['name']] = []
      separated[item['name']].append(item)

    for name in separated:
      separated[name] = downsampler(separated[name], 100)

    callback(json.dumps(list(chain(*separated.values()))))

if __name__ == '__main__':
  parse_command_line()

  __log__ = Log(options.log_directory)

  __application__ = Application(
    [
      (r'/visualization/(weights|activations)/(.*)', VisualizationHandler),
      (r'/inspection', InspectionHandler),
      (r'/loss', LossHandler),
      (r'/scores', ScoreHandler),
      (r'/(.*)', StaticFileHandler, {'path':options.static_directory}),
    ],
    debug=True,
  )

  __application__.listen(options.port)
  PeriodicCallback(__log__._update_records, 1000).start()
  IOLoop.instance().start()
