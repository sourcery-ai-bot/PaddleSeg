# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from qtpy import QtWidgets
from qtpy.QtCore import Qt


class TableWidget(QtWidgets.QTableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

    def dropEvent(self, event):
        if event.source() != self:
            return
        rows = {mi.row() for mi in self.selectedIndexes()}
        targetRow = self.indexAt(event.pos()).row()
        rows.discard(targetRow)
        rows = sorted(rows)
        if not rows:
            return
        if targetRow == -1:
            targetRow = self.rowCount()
        for _ in range(len(rows)):
            self.insertRow(targetRow)
        rowMapping = {}
        for idx, row in enumerate(rows):
            if row < targetRow:
                rowMapping[row] = targetRow + idx
            else:
                rowMapping[row + len(rows)] = targetRow + idx
        colCount = self.columnCount()
        for srcRow, tgtRow in sorted(rowMapping.items()):
            for col in range(colCount):
                self.setItem(tgtRow, col, self.takeItem(srcRow, col))
        for row in reversed(sorted(rowMapping.keys())):
            self.removeRow(row)
        event.accept()
        return

    def drop_on(self, event):
        index = self.indexAt(event.pos())
        if not index.isValid():
            return self.rowCount()

        return index.row() + 1 if self.is_below(event.pos(),
                                                index) else index.row()

    def is_below(self, pos, index):
        rect = self.visualRect(index)
        margin = 2
        if pos.y() - rect.top() < margin:
            return False
        elif rect.bottom() - pos.y() < margin:
            return True
        # noinspection PyTypeChecker
        return (rect.contains(pos, True) and
                not (int(self.model().flags(index)) & Qt.ItemIsDropEnabled) and
                pos.y() >= rect.center().y())
